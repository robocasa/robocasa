"""
This file contains several utility functions for working with environment
wrappers provided by the repository, and with environment metadata saved
in dataset files.
"""
from copy import deepcopy

# import robomimic.envs.env_base as EB
# from robomimic.utils.log_utils import log_warning


def get_env_class(env_meta=None, env_type=None, env=None):
    """
    Return env class from either env_meta, env_type, or env.
    Note the use of lazy imports - this ensures that modules are only
    imported when the corresponding env type is requested. This can
    be useful in practice. For example, a training run that only
    requires access to gym environments should not need to import
    robosuite.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env (instance of EB.EnvBase): environment instance
    """
    env_type = get_env_type(env_meta=env_meta, env_type=env_type, env=env)
    assert env_type == 1
    from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa

    return EnvRobocasa


def get_env_type(env_meta=None, env_type=None, env=None):
    """
    Helper function to get env_type from a variety of inputs.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env (instance of EB.EnvBase): environment instance
    """
    checks = [(env_meta is not None), (env_type is not None), (env is not None)]
    assert sum(checks) == 1, "should provide only one of env_meta, env_type, env"
    if env_meta is not None:
        env_type = env_meta["type"]
    elif env is not None:
        env_type = env.type
    return env_type


def check_env_type(type_to_check, env_meta=None, env_type=None, env=None):
    """
    Checks whether the passed env_meta, env_type, or env is of type @type_to_check.
    Type corresponds to EB.EnvType.

    Args:
        type_to_check (int): type to check equality against

        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env (instance of EB.EnvBase): environment instance
    """
    env_type = get_env_type(env_meta=env_meta, env_type=env_type, env=env)
    return env_type == type_to_check


def check_env_version(env, env_meta):
    """
    Checks whether the passed env and env_meta dictionary having matching environment versions.
    Logs warning if cannot find version or versions do not match.

    Args:
        env (instance of EB.EnvBase): environment instance

        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains following key:

                :`'env_version'`: environment version, type str
    """
    env_system_version = env.version
    env_meta_version = env_meta.get("env_version", None)

    # if env_meta_version is None:
    #     log_warning(
    #         "No environment version found in dataset!"\
    #         "\nCannot verify if dataset and installed environment versions match"\
    #     )
    # elif env_system_version != env_meta_version:
    #     log_warning(
    #         "Dataset and installed environment version mismatch!"\
    #         "\nDataset environment version: {meta}"\
    #         "\nInstalled environment version: {sys}".format(
    #             sys=env_system_version,
    #             meta=env_meta_version,
    #         )
    #     )


# def is_robosuite_env(env_meta=None, env_type=None, env=None):
#     """
#     Determines whether the environment is a robosuite environment. Accepts
#     either env_meta, env_type, or env.
#     """
#     return check_env_type(type_to_check=1, env_meta=env_meta, env_type=env_type, env=env)


def create_env(
    env_type,
    env_name,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
    env_lang=None,
    **kwargs,
):
    """
    Create environment.

    Args:
        env_type (int): the type of environment, which determines the env class that will
            be instantiated. Should be a value in EB.EnvType.

        env_name (str): name of environment

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if @use_image_obs is True.

        use_image_obs (bool): if True, environment is expected to render rgb image observations
            on every env.step call. Set this to False for efficiency reasons, if image
            observations are not required.

        env_lang: TODO documentation
    """

    # note: pass @postprocess_visual_obs True, to make sure images are processed for network inputs
    env_class = get_env_class(env_type=env_type)
    env = env_class(
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        postprocess_visual_obs=True,
        env_lang=env_lang,
        **kwargs,
    )
    print("Created environment with name {}".format(env_name))
    # print("Action size is {}".format(env.action_dimension))
    return env


def create_env_from_metadata(
    env_meta,
    env_name=None,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
    seed=None,
):
    """
    Create environment.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        env_name (str): name of environment. Only needs to be provided if making a different
            environment from the one in @env_meta.

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if @use_image_obs is True.

        use_image_obs (bool): if True, environment is expected to render rgb image observations
            on every env.step call. Set this to False for efficiency reasons, if image
            observations are not required.
    """
    if env_name is None:
        env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_name
    env_lang = env_meta.get("env_lang", None)

    if seed is not None:
        env_kwargs["seed"] = seed

    env = create_env(
        env_type=env_type,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        env_lang=env_lang,
        **env_kwargs,
    )
    check_env_version(env, env_meta)
    return env


def create_env_for_data_processing(
    env_meta,
    camera_names,
    camera_height,
    camera_width,
    reward_shaping,
    seed=None,
):
    """
    Creates environment for processing dataset observations and rewards.

    Args:
        env_meta (dict): environment metadata, which should be loaded from demonstration
            hdf5 with @FileUtils.get_env_metadata_from_dataset or from checkpoint (see
            @FileUtils.env_from_checkpoint). Contains 3 keys:

                :`'env_name'`: name of environment
                :`'type'`: type of environment, should be a value in EB.EnvType
                :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor

        camera_names (list of st): list of camera names that correspond to image observations

        camera_height (int): camera height for all cameras

        camera_width (int): camera width for all cameras

        reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
    """
    env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_class = get_env_class(env_type=env_type)

    # remove possibly redundant values in kwargs
    env_kwargs = deepcopy(env_kwargs)
    env_kwargs.pop("env_name", None)
    env_kwargs.pop("camera_names", None)
    env_kwargs.pop("camera_height", None)
    env_kwargs.pop("camera_width", None)
    env_kwargs.pop("reward_shaping", None)

    if seed is not None:
        env_kwargs["seed"] = seed

    env = env_class.create_for_data_processing(
        env_name=env_name,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=reward_shaping,
        **env_kwargs,
    )
    check_env_version(env, env_meta)
    return env


def wrap_env_from_config(env, config):
    """
    Wraps environment using the provided Config object to determine which wrappers
    to use (if any).
    """
    if config.train.frame_stack > 1:
        from robomimic.envs.wrappers import FrameStackWrapper

        env = FrameStackWrapper(env, num_frames=config.train.frame_stack)

    return env
