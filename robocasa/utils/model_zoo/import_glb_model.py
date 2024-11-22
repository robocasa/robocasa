import os
import shutil
from distutils.dir_util import copy_tree
import tempfile

from pathlib import Path

# import robosuite_model_zoo
import robocasa.utils.model_zoo.utils.mjcf_gen_utils as MJCFGenUtils
import robocasa.utils.model_zoo.utils.file_utils as FileUtils
import robocasa.utils.model_zoo.utils.parser_utils as ParserUtils
import robocasa.utils.model_zoo.utils.log_utils as LogUtils

# import robosuite_model_zoo.macros as macros

if __name__ == "__main__":
    parser = ParserUtils.get_base_parser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to existing (converted) objaverse model folder",
    )
    parser.add_argument(
        "--rot",
        type=str,
        required=False,
        help="axis to rotate around",
        choices=[
            "x",
            "y",
            "z",  # 90 degrees
            "x90",
            "x180",
            "x270",
            "y90",
            "y180",
            "y270",
            "z60",
            "z90",
            "z180",
            "z270",
        ],
        default="x",
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
    parser.add_argument(
        "--no_cached_coll",
        action="store_true",
        help="disable using cached coll geoms already generated in previous runs",
    )
    parser.add_argument("--center", type=str, nargs="?", const="False", default="True")
    parser.add_argument(
        "--prescale", type=str, nargs="?", const="False", default="True"
    )
    args = parser.parse_args()

    # if not os.path.isabs(args.path):
    #     args.path = os.path.join(macros.OBJAVERSE_PATH, args.path)

    verbose = args.verbose

    # get model name and asset path
    model_name = args.model_name or FileUtils.get_stem(args.path)[:8]
    # base_asset_path = args.base_asset_path or os.path.join(
    #     os.path.dirname(robosuite_model_zoo.__file__),
    #     "assets_private/objaverse",
    # )
    # asset_path = FileUtils.make_asset_path(
    #     base_asset_path,

    #     args.model_folder,
    #     model_name
    # )
    # asset_path = args.path

    # # save metadata
    # LogUtils.save_meta(args, asset_path)

    # # copy files to seprate temporary directories for collision / visual
    # tmp_model_path = tempfile.TemporaryDirectory()
    # tmp_coll_model_path = tempfile.TemporaryDirectory()

    # model_path = os.path.join(tmp_model_path.name, "models")
    # image_path = os.path.join(tmp_model_path.name, "images")
    # coll_model_path = tmp_coll_model_path.name

    # copy_tree(os.path.join(args.path, "models"), model_path)

    # if os.path.exists(os.path.join(args.path, "images")):
    #     copy_tree(os.path.join(args.path, "images"), image_path)

    # coll_objaverse_path = os.path.join(args.path, "collision")

    # if not Path(coll_objaverse_path).exists() or args.no_cached_coll:
    #     if os.path.exists(coll_objaverse_path):
    #         shutil.rmtree(coll_objaverse_path)

    #     MJCFGenUtils.decompose_convex(
    #         Path(os.path.join(args.path, "models/model_normalized.obj")),
    #         Path(coll_objaverse_path),

    #         max_output_convex_hulls=args.vhacd_max_output_convex_hulls,
    #         voxel_resolution=args.vhacd_voxel_resolution,
    #         volume_error_percent=args.vhacd_volume_error_percent,
    #         max_hull_vert_count=args.vhacd_max_hull_vert_count,
    #     )

    # copy_tree(coll_objaverse_path, coll_model_path)

    import trimesh

    glb_mesh = trimesh.load(os.path.join(args.path, "model.glb"))
    if not os.path.exists(os.path.join(args.path, "raw")):
        os.makedirs(os.path.join(args.path, "raw"))
    glb_mesh.export(os.path.join(args.path, "raw", "model_normalized.obj"))

    coll_path = os.path.join(args.path, "collision")
    if not Path(coll_path).exists() or args.no_cached_coll:
        MJCFGenUtils.decompose_convex(
            Path(os.path.join(args.path, "raw/model_normalized.obj")),
            Path(coll_path),
            max_output_convex_hulls=args.vhacd_max_output_convex_hulls,
            voxel_resolution=args.vhacd_voxel_resolution,
            volume_error_percent=args.vhacd_volume_error_percent,
            max_hull_vert_count=args.vhacd_max_hull_vert_count,
        )

    # get geom meta data
    model_info = MJCFGenUtils.parse_model_info(
        model_path=os.path.join(args.path, "raw"),
        model_name=model_name,
        coll_model_path=os.path.join(args.path, "collision"),
        asset_path=os.path.join(args.path),
        rot=args.rot,
        verbose=verbose,
        prescale=args.prescale,
        center=args.center,
    )

    # generate mjcf file
    mjcf_path = MJCFGenUtils.generate_mjcf(
        asset_path=os.path.join(args.path),
        model_name=model_name,
        model_info=model_info,
        sc=args.scale,
        hide_vis_geoms=args.hide_vis_geoms,
        show_coll_geoms=args.show_coll_geoms,
        show_sites=args.show_sites,
        verbose=verbose,
    )

    # # copy over raw data
    # raw_folder = os.path.join(asset_path, "raw")
    # os.makedirs(raw_folder)
    # copy_tree(
    #     os.path.join(args.path, "models"),
    #     raw_folder
    # )

    # # delete temp directories
    # tmp_model_path.cleanup()
    # tmp_coll_model_path.cleanup()
