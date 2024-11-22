import numpy as np
import os
import xml.etree.ElementTree as ET

import robocasa.utils.model_zoo.utils.mjcf_gen_utils as MJCFGenUtils
import robocasa.utils.model_zoo.utils.file_utils as FileUtils
import robocasa.utils.model_zoo.utils.parser_utils as ParserUtils
import robocasa.utils.model_zoo.utils.log_utils as LogUtils


if __name__ == "__main__":
    parser = ParserUtils.get_base_parser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to model folder containing .obj and .mtl files",
    )
    parser.add_argument(
        "--coll_model_path",
        type=str,
        help="(optional) path to collision .obj file. "
        "If specified, --model_path only serves as a visual model. "
        "Note: collision model must only consist of primitive shapes "
        "(cubes, cylinders, spheres). ",
    )
    parser.add_argument(
        "--texture_path",
        type=str,
        help="path to .png corresponding to texture used in --model_path",
    )
    parser.add_argument(
        "--show_coll_geoms",
        action="store_true",
        help="visually show collision geoms in addition to visual geoms",
    )
    parser.add_argument(
        "--hide_vis_geoms",
        action="store_true",
        help="hide visual geoms",
    )

    args = parser.parse_args()

    model_path = args.model_path
    coll_model_path = args.coll_model_path
    texture_path = args.texture_path

    verbose = args.verbose

    # process model_path
    assert model_path is not None
    model_path = os.path.expanduser(model_path)
    assert os.path.exists(model_path)

    # get model name and asset path
    model_name = args.model_name
    base_asset_path = args.base_asset_path

    assert model_name is not None

    # get model name and asset path
    model_name = args.model_name
    if model_name is None:
        model_name = FileUtils.get_stem(args.path)
    asset_path = FileUtils.get_asset_path(
        args.base_asset_path, args.model_folder, model_name
    )

    # save metadata
    LogUtils.save_meta(args, asset_path)

    # new_model_path = os.path.join(asset_path, "visual")

    LogUtils.maybe_log_info("Copying assets to {}".format(asset_path), verbose)
    # if new_model_path != model_path:
    #     if os.path.exists(new_model_path):
    #         shutil.rmtree(new_model_path)
    #     shutil.copytree(
    #         model_path,
    #         new_model_path
    #     )

    # process coll_model_path
    if coll_model_path is not None:
        # raise NotImplementedError
        coll_model_path = os.path.expanduser(coll_model_path)
        assert os.path.exists(coll_model_path)
        # coll_model_name, coll_model_type = get_file_name_and_type(coll_model_path)
        # assert coll_model_type == "obj", (
        #     "--coll_model_path must point to .obj file"
        # )
        # paths_to_copy.append(coll_model_path)

    # process texture_path
    if texture_path is not None:
        raise NotImplementedError
        texture_path = os.path.expanduser(texture_path)
        assert os.path.exists(texture_path)
        paths_to_copy.append(texture_path)
        # texture_path points to new location in asset folder
        texture_path = os.path.join(asset_path, os.path.basename(texture_path))

    # # copy asset files to asset folder
    # LogUtils.maybe_log_info("Copying assets to {}".format(asset_path), verbose)
    # for path in paths_to_copy:
    #     if os.path.exists(path):
    #         shutil.copyfile(
    #             path,
    #             os.path.join(asset_path, os.path.basename(path))
    #         )

    geoms = MJCFGenUtils.parse_geoms(
        model_path=model_path,
        model_name=model_name,
        coll_model_path=coll_model_path,
        asset_path=asset_path,
        verbose=verbose,
    )

    mjcf_path = MJCFGenUtils.generate_mjcf(
        asset_path=asset_path,
        model_name=model_name,
        geoms=geoms,
        sc=args.scale,
        texture_path=texture_path,
        hide_vis_geoms=args.hide_vis_geoms,
        show_coll_geoms=args.show_coll_geoms,
        verbose=verbose,
    )
