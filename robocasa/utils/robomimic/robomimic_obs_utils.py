"""
A collection of utilities for working with observation dictionaries and
different kinds of modalities such as images.
"""
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F

import robocasa.utils.robomimic.robomimic_tensor_utils as TU

# MACRO FOR VALID IMAGE CHANNEL SIZES
VALID_IMAGE_CHANNEL_DIMS = {1, 3}  # depth, rgb

# DO NOT MODIFY THIS!
# This keeps track of observation types (modalities) - and is populated on call to @initialize_obs_utils_with_obs_specs.
# This will be a dictionary that maps observation modality (e.g. low_dim, rgb) to a list of observation
# keys under that observation modality.
OBS_MODALITIES_TO_KEYS = None

# DO NOT MODIFY THIS!
# This keeps track of observation types (modalities) - and is populated on call to @initialize_obs_utils_with_obs_specs.
# This will be a dictionary that maps observation keys to their corresponding observation modality
# (e.g. low_dim, rgb)
OBS_KEYS_TO_MODALITIES = None

# DO NOT MODIFY THIS
# This holds the default encoder kwargs that will be used if none are passed at runtime for any given network
DEFAULT_ENCODER_KWARGS = None

# DO NOT MODIFY THIS
# This holds the registered observation modality classes
OBS_MODALITY_CLASSES = {}

# DO NOT MODIFY THIS
# This global dict stores mapping from observation encoder / randomizer network name to class.
# We keep track of these registries to enable automated class inference at runtime, allowing
# users to simply extend our base encoder / randomizer class and refer to that class in string form
# in their config, without having to manually register their class internally.
# This also future-proofs us for any additional encoder / randomizer classes we would
# like to add ourselves.
OBS_ENCODER_CORES = {"None": None}  # Include default None
OBS_RANDOMIZERS = {"None": None}  # Include default None


def register_obs_key(target_class):
    assert (
        target_class not in OBS_MODALITY_CLASSES
    ), f"Already registered modality {target_class}!"
    OBS_MODALITY_CLASSES[target_class.name] = target_class


def register_encoder_core(target_class):
    assert (
        target_class not in OBS_ENCODER_CORES
    ), f"Already registered obs encoder core {target_class}!"
    OBS_ENCODER_CORES[target_class.__name__] = target_class


def register_randomizer(target_class):
    assert (
        target_class not in OBS_RANDOMIZERS
    ), f"Already registered obs randomizer {target_class}!"
    OBS_RANDOMIZERS[target_class.__name__] = target_class


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def obs_encoder_kwargs_from_config(obs_encoder_config):
    """
    Generate a set of args used to create visual backbones for networks
    from the observation encoder config.

    Args:
        obs_encoder_config (Config): Config object containing relevant encoder information. Should be equivalent to
            config.observation.encoder

    Returns:
        dict: Processed encoder kwargs
    """
    # Loop over each obs modality
    # Unlock encoder config
    obs_encoder_config.unlock()
    for obs_modality, encoder_kwargs in obs_encoder_config.items():
        ###  TODO: fix sanity checks. Disabling this code snippet to allow chaining multiple randomizers.
        # # First run some sanity checks and store the classes
        # for cls_name, cores in zip(("core", "obs_randomizer"), (OBS_ENCODER_CORES, OBS_RANDOMIZERS)):
        #     # Make sure the requested encoder for each obs_modality exists
        #     cfg_cls = encoder_kwargs[f"{cls_name}_class"]
        #     if cfg_cls is not None:
        #         assert cfg_cls in cores, f"No {cls_name} class with name {cfg_cls} found, must register this class before" \
        #             f"creating model!"
        #         # encoder_kwargs[f"{cls_name}_class"] = cores[cfg_cls]

        # Process core and randomizer kwargs
        encoder_kwargs.core_kwargs = (
            dict()
            if encoder_kwargs.core_kwargs is None
            else deepcopy(encoder_kwargs.core_kwargs)
        )
        encoder_kwargs.obs_randomizer_kwargs = (
            dict()
            if encoder_kwargs.obs_randomizer_kwargs is None
            else deepcopy(encoder_kwargs.obs_randomizer_kwargs)
        )

    # Re-lock keys
    obs_encoder_config.lock()

    return dict(obs_encoder_config)


def initialize_obs_modality_mapping_from_dict(modality_mapping):
    """
    This function is an alternative to @initialize_obs_utils_with_obs_specs, that allows manually setting of modalities.
    NOTE: Only one of these should be called at runtime -- not both! (Note that all training scripts that use a config)
        automatically handle obs modality mapping, so using this function is usually unnecessary)

    Args:
        modality_mapping (dict): Maps modality string names (e.g.: rgb, low_dim, etc.) to a list of observation
            keys that should belong to that modality
    """
    global OBS_KEYS_TO_MODALITIES, OBS_MODALITIES_TO_KEYS

    OBS_KEYS_TO_MODALITIES = ObservationKeyToModalityDict()
    OBS_MODALITIES_TO_KEYS = dict()

    for mod, keys in modality_mapping.items():
        OBS_MODALITIES_TO_KEYS[mod] = deepcopy(keys)
        OBS_KEYS_TO_MODALITIES.update({k: mod for k in keys})


def initialize_obs_utils_with_obs_specs(obs_modality_specs):
    """
    This function should be called before using any observation key-specific
    functions in this file, in order to make sure that all utility
    functions are aware of the observation modalities (e.g. which ones
    are low-dimensional, which ones are rgb, etc.).

    It constructs two dictionaries: (1) that map observation modality (e.g. low_dim, rgb) to
    a list of observation keys under that modality, and (2) that maps the inverse, specific
    observation keys to their corresponding observation modality.

    Input should be a nested dictionary (or list of such dicts) with the following structure:

        obs_variant (str):
            obs_modality (str): observation keys (list)
            ...
        ...

    Example:
        {
            "obs": {
                "low_dim": ["robot0_eef_pos", "robot0_eef_quat"],
                "rgb": ["agentview_image", "robot0_eye_in_hand"],
            }
            "goal": {
                "low_dim": ["robot0_eef_pos"],
                "rgb": ["agentview_image"]
            }
        }

    In the example, raw observations consist of low-dim and rgb modalities, with
    the robot end effector pose under low-dim, and the agentview and wrist camera
    images under rgb, while goal observations also consist of low-dim and rgb modalities,
    with a subset of the raw observation keys per modality.

    Args:
        obs_modality_specs (dict or list): A nested dictionary (see docstring above for an example)
            or a list of nested dictionaries. Accepting a list as input makes it convenient for
            situations where multiple modules may each have their own modality spec.
    """
    global OBS_KEYS_TO_MODALITIES, OBS_MODALITIES_TO_KEYS

    OBS_KEYS_TO_MODALITIES = ObservationKeyToModalityDict()

    # accept one or more spec dictionaries - if it's just one, account for this
    if isinstance(obs_modality_specs, dict):
        obs_modality_spec_list = [obs_modality_specs]
    else:
        obs_modality_spec_list = obs_modality_specs

    # iterates over observation specs
    obs_modality_mapping = {}
    for obs_modality_spec in obs_modality_spec_list:
        # iterates over observation variants (e.g. observations, goals, subgoals)
        for obs_modalities in obs_modality_spec.values():
            for obs_modality, obs_keys in obs_modalities.items():
                # add all keys for each obs modality to the corresponding list in obs_modality_mapping
                if obs_modality not in obs_modality_mapping:
                    obs_modality_mapping[obs_modality] = []
                obs_modality_mapping[obs_modality] += obs_keys
                # loop over each modality, and add to global dict if it doesn't exist yet
                for obs_key in obs_keys:
                    if obs_key not in OBS_KEYS_TO_MODALITIES:
                        OBS_KEYS_TO_MODALITIES[obs_key] = obs_modality
                    # otherwise, run sanity check to make sure we don't have conflicting, duplicate entries
                    else:
                        assert OBS_KEYS_TO_MODALITIES[obs_key] == obs_modality, (
                            f"Cannot register obs key {obs_key} with modality {obs_modality}; "
                            f"already exists with corresponding modality {OBS_KEYS_TO_MODALITIES[obs_key]}"
                        )

    # remove duplicate entries and store in global mapping
    OBS_MODALITIES_TO_KEYS = {
        obs_modality: list(set(obs_modality_mapping[obs_modality]))
        for obs_modality in obs_modality_mapping
    }

    # print("\n============= Initialized Observation Utils with Obs Spec =============\n")
    # for obs_modality, obs_keys in OBS_MODALITIES_TO_KEYS.items():
    #     print("using obs modality: {} with keys: {}".format(obs_modality, obs_keys))


def initialize_default_obs_encoder(obs_encoder_config):
    """
    Initializes the default observation encoder kwarg information to be used by all networks if no values are manually
    specified at runtime.

    Args:
        obs_encoder_config (Config): Observation encoder config to use.
            Should be equivalent to config.observation.encoder
    """
    global DEFAULT_ENCODER_KWARGS
    DEFAULT_ENCODER_KWARGS = obs_encoder_kwargs_from_config(obs_encoder_config)


def initialize_obs_utils_with_config(config):
    """
    Utility function to parse config and call @initialize_obs_utils_with_obs_specs and
    @initialize_default_obs_encoder_kwargs with the correct arguments.

    Args:
        config (BaseConfig instance): config object
    """
    if config.algo_name == "hbc":
        obs_modality_specs = [
            config.observation.planner.modalities,
            config.observation.actor.modalities,
        ]
        obs_encoder_config = config.observation.actor.encoder
    elif config.algo_name == "iris":
        obs_modality_specs = [
            config.observation.value_planner.planner.modalities,
            config.observation.value_planner.value.modalities,
            config.observation.actor.modalities,
        ]
        obs_encoder_config = config.observation.actor.encoder
    else:
        obs_modality_specs = [config.observation.modalities]
        obs_encoder_config = config.observation.encoder
    initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_modality_specs)
    initialize_default_obs_encoder(obs_encoder_config=obs_encoder_config)


def key_is_obs_modality(key, obs_modality):
    """
    Check if observation key corresponds to modality @obs_modality.

    Args:
        key (str): obs key name to check
        obs_modality (str): observation modality - e.g.: "low_dim", "rgb"
    """
    assert (
        OBS_KEYS_TO_MODALITIES is not None
    ), "error: must call ObsUtils.initialize_obs_utils_with_obs_config first"
    return OBS_KEYS_TO_MODALITIES[key] == obs_modality


def center_crop(im, t_h, t_w):
    """
    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]


def batch_image_hwc_to_chw(im):
    """
    Channel swap for images - useful for preparing images for
    torch training.

    Args:
        im (np.array or torch.Tensor): image of shape (batch, height, width, channel)
            or (height, width, channel)

    Returns:
        im (np.array or torch.Tensor): image of shape (batch, channel, height, width)
            or (channel, height, width)
    """
    start_dims = np.arange(len(im.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    if isinstance(im, np.ndarray):
        return im.transpose(start_dims + [s + 3, s + 1, s + 2])
    else:
        return im.permute(start_dims + [s + 3, s + 1, s + 2])


def batch_image_chw_to_hwc(im):
    """
    Inverse of channel swap in @batch_image_hwc_to_chw.

    Args:
        im (np.array or torch.Tensor): image of shape (batch, channel, height, width)
            or (channel, height, width)

    Returns:
        im (np.array or torch.Tensor): image of shape (batch, height, width, channel)
            or (height, width, channel)
    """
    start_dims = np.arange(len(im.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    if isinstance(im, np.ndarray):
        return im.transpose(start_dims + [s + 2, s + 3, s + 1])
    else:
        return im.permute(start_dims + [s + 2, s + 3, s + 1])


def process_obs(obs, obs_modality=None, obs_key=None):
    """
    Process observation @obs corresponding to @obs_modality modality (or implicitly inferred from @obs_key)
    to prepare for network input.

    Note that either obs_modality OR obs_key must be specified!

    If both are specified, obs_key will override obs_modality

    Args:
        obs (np.array or torch.Tensor): Observation to process. Leading batch dimension is optional
        obs_modality (str): Observation modality (e.g.: depth, image, low_dim, etc.)
        obs_key (str): Name of observation from which to infer @obs_modality

    Returns:
        processed_obs (np.array or torch.Tensor): processed observation
    """
    assert (
        obs_modality is not None or obs_key is not None
    ), "Either obs_modality or obs_key must be specified!"
    if obs_key is not None:
        obs_modality = OBS_KEYS_TO_MODALITIES[obs_key]
    return OBS_MODALITY_CLASSES[obs_modality].process_obs(obs)


def process_obs_dict(obs_dict):
    """
    Process observations in observation dictionary to prepare for network input.

    Args:
        obs_dict (dict): dictionary mapping observation keys to np.array or
            torch.Tensor. Leading batch dimensions are optional.

    Returns:
        new_dict (dict): dictionary where observation keys have been processed by their corresponding processors
    """
    return {
        k: process_obs(obs=obs, obs_key=k) for k, obs in obs_dict.items()
    }  # shallow copy


def process_frame(frame, channel_dim, scale):
    """
    Given frame fetched from dataset, process for network input. Converts array
    to float (from uint8), normalizes pixels from range [0, @scale] to [0, 1], and channel swaps
    from (H, W, C) to (C, H, W).

    Args:
        frame (np.array or torch.Tensor): frame array
        channel_dim (int): Number of channels to sanity check for
        scale (float): Value to normalize inputs by

    Returns:
        processed_frame (np.array or torch.Tensor): processed frame
    """
    # Channel size should either be 3 (RGB) or 1 (depth)
    assert frame.shape[-1] == channel_dim
    frame = TU.to_float(frame)
    frame /= scale
    frame = frame.clip(0.0, 1.0)
    frame = batch_image_hwc_to_chw(frame)

    return frame


def unprocess_obs(obs, obs_modality=None, obs_key=None):
    """
    Prepare observation @obs corresponding to @obs_modality modality (or implicitly inferred from @obs_key)
    to prepare for deployment.

    Note that either obs_modality OR obs_key must be specified!

    If both are specified, obs_key will override obs_modality

    Args:
        obs (np.array or torch.Tensor): Observation to unprocess. Leading batch dimension is optional
        obs_modality (str): Observation modality (e.g.: depth, image, low_dim, etc.)
        obs_key (str): Name of observation from which to infer @obs_modality

    Returns:
        unprocessed_obs (np.array or torch.Tensor): unprocessed observation
    """
    assert (
        obs_modality is not None or obs_key is not None
    ), "Either obs_modality or obs_key must be specified!"
    if obs_key is not None:
        obs_modality = OBS_KEYS_TO_MODALITIES[obs_key]
    return OBS_MODALITY_CLASSES[obs_modality].unprocess_obs(obs)


def unprocess_obs_dict(obs_dict):
    """
    Prepare processed observation dictionary for saving to dataset. Inverse of
    @process_obs.

    Args:
        obs_dict (dict): dictionary mapping observation keys to np.array or
            torch.Tensor. Leading batch dimensions are optional.

    Returns:
        new_dict (dict): dictionary where observation keys have been unprocessed by
            their respective unprocessor methods
    """
    return {
        k: unprocess_obs(obs=obs, obs_key=k) for k, obs in obs_dict.items()
    }  # shallow copy


def unprocess_frame(frame, channel_dim, scale):
    """
    Given frame prepared for network input, prepare for saving to dataset.
    Inverse of @process_frame.

    Args:
        frame (np.array or torch.Tensor): frame array
        channel_dim (int): What channel dimension should be (used for sanity check)
        scale (float): Scaling factor to apply during denormalization

    Returns:
        unprocessed_frame (np.array or torch.Tensor): frame passed through
            inverse operation of @process_frame
    """
    assert frame.shape[-3] == channel_dim  # check for channel dimension
    frame = batch_image_chw_to_hwc(frame)
    frame *= scale
    return frame


def get_processed_shape(obs_modality, input_shape):
    """
    Given observation modality @obs_modality and expected inputs of shape @input_shape (excluding batch dimension), return the
    expected processed observation shape resulting from process_{obs_modality}.

    Args:
        obs_modality (str): Observation modality to use (e.g.: low_dim, rgb, depth, etc...)
        input_shape (list of int): Expected input dimensions, excluding the batch dimension

    Returns:
        list of int: expected processed input shape
    """
    return list(process_obs(obs=np.zeros(input_shape), obs_modality=obs_modality).shape)


def normalize_dict(dict, normalization_stats):
    """
    Normalize dict using the provided "offset" and "scale" entries
    for each observation key. The dictionary will be
    modified in-place.

    Args:
        dict (dict): dictionary mapping key to np.array or
            torch.Tensor. Leading batch dimensions are optional.

        normalization_stats (dict): this should map keys to dicts
            with a "offset" and "scale" of shape (1, ...) where ... is the default
            shape for the dict value.

    Returns:
        dict (dict): obs dict with normalized arrays
    """

    # ensure we have statistics for each modality key in the dict
    assert set(dict.keys()).issubset(normalization_stats)

    for m in dict:
        offset = normalization_stats[m]["offset"]
        scale = normalization_stats[m]["scale"]

        # check shape consistency
        shape_len_diff = len(offset.shape) - len(dict[m].shape)
        assert shape_len_diff in [0, 1], "shape length mismatch in @normalize_dict"
        # if dict has no leading batch dim, check shapes match exactly, else allow first dim to broadcast
        assert (
            offset.shape[1:] == dict[m].shape[(1 - shape_len_diff) :]
        ), "shape mismatch in @normalize_dict"

        # handle case where obs dict is not batched by removing stats batch dimension
        if shape_len_diff == 1:
            offset = offset[0]
            scale = scale[0]

        dict[m] = (dict[m] - offset) / scale

    return dict


def unnormalize_dict(dict, normalization_stats):
    """
    Unnormalize dict using the provided "offset" and "scale" entries
    for each observation key. The dictionary will be
    modified in-place.

    Args:
        dict (dict): dictionary mapping key to np.array or
            torch.Tensor. Leading batch dimensions are optional.

        normalization_stats (dict): this should map keys to dicts
            with a "offset" and "scale" of shape (1, ...) where ... is the default
            shape for the dict value.

    Returns:
        dict (dict): obs dict with normalized arrays
    """

    # ensure we have statistics for each modality key in the dict
    assert set(dict.keys()).issubset(normalization_stats)

    for m in dict:
        offset = normalization_stats[m]["offset"]
        scale = normalization_stats[m]["scale"]

        # check shape consistency
        shape_len_diff = len(offset.shape) - len(dict[m].shape)
        assert shape_len_diff in [0, 1], "shape length mismatch in @unnormalize_dict"
        # if dict has no leading batch dim, check shapes match exactly, else allow first dim to broadcast
        assert (
            offset.shape[1:] == dict[m].shape[(1 - shape_len_diff) :]
        ), "shape mismatch in @unnormalize_dict"

        # handle case where obs dict is not batched by removing stats batch dimension
        if shape_len_diff == 1:
            offset = offset[0]
            scale = scale[0]

        dict[m] = (dict[m] * scale) + offset

    return dict


def has_modality(modality, obs_keys):
    """
    Returns True if @modality is present in the list of observation keys @obs_keys.

    Args:
        modality (str): modality to check for, e.g.: rgb, depth, etc.
        obs_keys (list): list of observation keys
    """
    for k in obs_keys:
        if key_is_obs_modality(k, obs_modality=modality):
            return True
    return False


def repeat_and_stack_observation(obs_dict, n):
    """
    Given an observation dictionary and a desired repeat value @n,
    this function will return a new observation dictionary where
    each modality is repeated @n times and the copies are
    stacked in the first dimension.

    For example, if a batch of 3 observations comes in, and n is 2,
    the output will look like [ob1; ob1; ob2; ob2; ob3; ob3] in
    each modality.

    Args:
        obs_dict (dict): dictionary mapping observation key to np.array or
            torch.Tensor. Leading batch dimensions are optional.

        n (int): number to repeat by

    Returns:
        repeat_obs_dict (dict): repeated obs dict
    """
    return TU.repeat_by_expand_at(obs_dict, repeats=n, dim=0)


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """
    Crops images at the locations specified by @crop_indices. Crops will be
    taken across all channels.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_indices (torch.Tensor): batch of indices of shape [..., N, 2] where
            N is the number of crops to take per image and each entry corresponds
            to the pixel height and width of where to take the crop. Note that
            the indices can also be of shape [..., 2] if only 1 crop should
            be taken per image. Leading dimensions must be consistent with
            @images argument. Each index specifies the top left of the crop.
            Values must be in range [0, H - CH - 1] x [0, W - CW - 1] where
            H and W are the height and width of @images and CH and CW are
            @crop_height and @crop_width.

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

    Returns:
        crops (torch.Tesnor): cropped images of shape [..., C, @crop_height, @crop_width]
    """

    # make sure length of input shapes is consistent
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (
        ndim_im_shape == ndim_indices_shape + 2
    )

    # maybe pad so that @crop_indices is shape [..., N, 2]
    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    # make sure leading dimensions between images and indices are consistent
    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    # make sure @crop_indices are in valid range
    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    # convert each crop index (ch, cw) into a list of pixel indices that correspond to the entire window.

    # 2D index array with columns [0, 1, ..., CH - 1] and shape [CH, CW]
    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = TU.unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    # 2D index array with rows [0, 1, ..., CW - 1] and shape [CH, CW]
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = TU.unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    # combine into shape [CH, CW, 2]
    crop_in_grid = torch.cat(
        (crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1
    )

    # Add above grid with the offset index of each sampled crop to get 2d indices for each crop.
    # After broadcasting, this will be shape [..., N, CH, CW, 2] and each crop has a [CH, CW, 2]
    # shape array that tells us which pixels from the corresponding source image to grab.
    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(
        grid_reshape
    )

    # For using @torch.gather, convert to flat indices from 2D indices, and also
    # repeat across the channel dimension. To get flat index of each pixel to grab for
    # each sampled crop, we just use the mapping: ind = h_ind * @image_w + w_ind
    all_crop_inds = (
        all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1]
    )  # shape [..., N, CH, CW]
    all_crop_inds = TU.unsqueeze_expand_at(
        all_crop_inds, size=image_c, dim=-3
    )  # shape [..., N, C, CH, CW]
    all_crop_inds = TU.flatten(
        all_crop_inds, begin_axis=-2
    )  # shape [..., N, C, CH * CW]

    # Repeat and flatten the source images -> [..., N, C, H * W] and then use gather to index with crop pixel inds
    images_to_crop = TU.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = TU.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    # [..., N, C, CH * CW] -> [..., N, C, CH, CW]
    reshape_axis = len(crops.shape) - 1
    crops = TU.reshape_dimensions(
        crops,
        begin_axis=reshape_axis,
        end_axis=reshape_axis,
        target_dims=(crop_height, crop_width),
    )

    if is_padded:
        # undo padding -> [..., C, CH, CW]
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(
    images, crop_height, crop_width, num_crops, pos_enc=False
):
    """
    For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

        num_crops (n): number of crops to sample

        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial
            encoding of the original source pixel locations. This means that the
            output crops will contain information about where in the source image
            it was sampled from.

    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width)
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)

        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device

    # maybe add 2 channels of spatial encoding to the source image
    source_im = images
    if pos_enc:
        # spatial encoding [y, x] in [0, 1]
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x))  # shape [C, H, W]

        # unsqueeze and expand to match leading dimensions -> shape [..., C, H, W]
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)

        # concat across channel dimension with input
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # make sure sample boundaries ensure crops are fully within the images
    image_c, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
    # Each gets @num_crops samples - typically this will just be the batch dimension (B), so
    # we will sample [B, N] indices, but this supports having more than one leading dimension,
    # or possibly no leading dimension.
    #
    # Trick: sample in [0, 1) with rand, then re-scale to [0, M) and convert to long to get sampled ints
    crop_inds_h = (
        max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)
    ).long()
    crop_inds_w = (
        max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)
    ).long()
    crop_inds = torch.cat(
        (crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1
    )  # shape [..., N, 2]

    crops = crop_image_from_indices(
        images=source_im,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )

    return crops, crop_inds


class Modality:
    """
    Observation Modality class to encapsulate necessary functions needed to
    process observations of this modality
    """

    # observation keys to associate with this modality
    keys = set()

    # Custom processing function that should prepare raw observations of this modality for training
    _custom_obs_processor = None

    # Custom unprocessing function that should prepare observations of this modality used during training for deployment
    _custom_obs_unprocessor = None

    # Name of this modality -- must be set by subclass!
    name = None

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid modalities
        """
        assert (
            cls.name is not None
        ), f"Name of modality {cls.__name__} must be specified!"
        register_obs_key(cls)

    @classmethod
    def set_keys(cls, keys):
        """
        Sets the observation keys associated with this modality.

        Args:
            keys (list or set): observation keys to associate with this modality
        """
        cls.keys = {k for k in keys}

    @classmethod
    def add_keys(cls, keys):
        """
        Adds the observation @keys associated with this modality to the current set of keys.

        Args:
            keys (list or set): observation keys to add to associate with this modality
        """
        for key in keys:
            cls.keys.add(key)

    @classmethod
    def set_obs_processor(cls, processor=None):
        """
        Sets the processor for this observation modality. If @processor is set to None, then
        the obs processor will use the default one (self.process_obs(...)). Otherwise, @processor
        should be a function to process this corresponding observation modality.

        Args:
            processor (function or None): If not None, should be function that takes in either a
                np.array or torch.Tensor and output the processed array / tensor. If None, will reset
                to the default processor (self.process_obs(...))
        """
        cls._custom_obs_processor = processor

    @classmethod
    def set_obs_unprocessor(cls, unprocessor=None):
        """
        Sets the unprocessor for this observation modality. If @unprocessor is set to None, then
        the obs unprocessor will use the default one (self.unprocess_obs(...)). Otherwise, @unprocessor
        should be a function to process this corresponding observation modality.

        Args:
            unprocessor (function or None): If not None, should be function that takes in either a
                np.array or torch.Tensor and output the unprocessed array / tensor. If None, will reset
                to the default unprocessor (self.unprocess_obs(...))
        """
        cls._custom_obs_unprocessor = unprocessor

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Default processing function for this obs modality.

        Note that this function is overridden by self.custom_obs_processor (a function with identical inputs / outputs)
        if it is not None.

        Args:
            obs (np.array or torch.Tensor): raw observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: processed observation
        """
        raise NotImplementedError

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Default unprocessing function for this obs modality.

        Note that this function is overridden by self.custom_obs_unprocessor
        (a function with identical inputs / outputs) if it is not None.

        Args:
            obs (np.array or torch.Tensor): processed observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: unprocessed observation
        """
        raise NotImplementedError

    @classmethod
    def process_obs(cls, obs):
        """
        Prepares an observation @obs of this modality for network input.

        Args:
            obs (np.array or torch.Tensor): raw observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: processed observation
        """
        processor = (
            cls._custom_obs_processor
            if cls._custom_obs_processor is not None
            else cls._default_obs_processor
        )
        return processor(obs)

    @classmethod
    def unprocess_obs(cls, obs):
        """
        Prepares an observation @obs of this modality for deployment.

        Args:
            obs (np.array or torch.Tensor): processed observation, which may include a leading batch dimension

        Returns:
            np.array or torch.Tensor: unprocessed observation
        """
        unprocessor = (
            cls._custom_obs_unprocessor
            if cls._custom_obs_unprocessor is not None
            else cls._default_obs_unprocessor
        )
        return unprocessor(obs)

    @classmethod
    def process_obs_from_dict(cls, obs_dict, inplace=True):
        """
        Receives a dictionary of keyword mapped observations @obs_dict, and processes the observations with keys
        corresponding to this modality. A copy will be made of the received dictionary unless @inplace is True

        Args:
            obs_dict (dict): Dictionary mapping observation keys to observations
            inplace (bool): If True, will modify @obs_dict in place, otherwise, will create a copy

        Returns:
            dict: observation dictionary with processed observations corresponding to this modality
        """
        if inplace:
            obs_dict = deepcopy(obs_dict)
        # Loop over all keys and process the ones corresponding to this modality
        for key, obs in obs_dict.values():
            if key in cls.keys:
                obs_dict[key] = cls.process_obs(obs)

        return obs_dict


class ImageModality(Modality):
    """
    Modality for RGB image observations
    """

    name = "rgb"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given image fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 255] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).

        Args:
            obs (np.array or torch.Tensor): image array

        Returns:
            processed_obs (np.array or torch.Tensor): processed image
        """
        return process_frame(frame=obs, channel_dim=3, scale=255.0)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given image prepared for network input, prepare for saving to dataset.
        Inverse of @process_frame.

        Args:
            obs (np.array or torch.Tensor): image array

        Returns:
            unprocessed_obs (np.array or torch.Tensor): image passed through
                inverse operation of @process_frame
        """
        return TU.to_uint8(unprocess_frame(frame=obs, channel_dim=3, scale=255.0))


class DepthModality(Modality):
    """
    Modality for depth observations
    """

    name = "depth"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given depth fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 1] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            processed_obs (np.array or torch.Tensor): processed depth
        """
        return process_frame(frame=obs, channel_dim=1, scale=1.0)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given depth prepared for network input, prepare for saving to dataset.
        Inverse of @process_depth.

        Args:
            obs (np.array or torch.Tensor): depth array

        Returns:
            unprocessed_obs (np.array or torch.Tensor): depth passed through
                inverse operation of @process_depth
        """
        return unprocess_frame(frame=obs, channel_dim=1, scale=1.0)


class ScanModality(Modality):
    """
    Modality for scan observations
    """

    name = "scan"

    @classmethod
    def _default_obs_processor(cls, obs):
        return obs

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        return obs


class LowDimModality(Modality):
    """
    Modality for low dimensional observations
    """

    name = "low_dim"

    @classmethod
    def _default_obs_processor(cls, obs):
        return obs

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        return obs
