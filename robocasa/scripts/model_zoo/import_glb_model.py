import os
import shutil
from distutils.dir_util import copy_tree
import trimesh
import robocasa
from termcolor import colored

from pathlib import Path

import robocasa.utils.model_zoo.mjcf_gen_utils as MJCFGenUtils
import robocasa.utils.model_zoo.file_utils as FileUtils
import robocasa.utils.model_zoo.parser_utils as ParserUtils
import robocasa.utils.model_zoo.log_utils as LogUtils

if __name__ == "__main__":
    parser = ParserUtils.get_base_parser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to existing .glb model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to store converted model",
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
    model_name = args.model_name or FileUtils.get_stem(args.path)
    output_path = args.output_path or os.path.join(
        os.path.dirname(robocasa.__file__), "models/assets/model_zoo_assets", model_name
    )
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # # save metadata
    # LogUtils.save_meta(args, asset_path)

    glb_mesh = trimesh.load(args.path)
    if not os.path.exists(os.path.join(output_path, "raw")):
        os.makedirs(os.path.join(output_path, "raw"))
    glb_mesh.export(os.path.join(output_path, "raw", "model_normalized.obj"))

    coll_path = os.path.join(output_path, "collision")
    if not Path(coll_path).exists() or args.no_cached_coll:
        MJCFGenUtils.decompose_convex(
            Path(os.path.join(output_path, "raw/model_normalized.obj")),
            Path(coll_path),
            max_output_convex_hulls=args.vhacd_max_output_convex_hulls,
            voxel_resolution=args.vhacd_voxel_resolution,
            volume_error_percent=args.vhacd_volume_error_percent,
            max_hull_vert_count=args.vhacd_max_hull_vert_count,
        )

    # get geom meta data
    model_info = MJCFGenUtils.parse_model_info(
        model_path=os.path.join(output_path, "raw"),
        model_name=model_name,
        coll_model_path=os.path.join(output_path, "collision"),
        asset_path=os.path.join(output_path),
        rot=args.rot,
        verbose=verbose,
        prescale=args.prescale,
        center=args.center,
    )

    # generate mjcf file
    mjcf_path = MJCFGenUtils.generate_mjcf(
        asset_path=os.path.join(output_path),
        model_name=model_name,
        model_info=model_info,
        sc=args.scale,
        hide_vis_geoms=args.hide_vis_geoms,
        show_coll_geoms=args.show_coll_geoms,
        show_sites=args.show_sites,
        verbose=verbose,
    )

    print(colored(f"Model output to: {output_path}/model.xml", color="green"))
