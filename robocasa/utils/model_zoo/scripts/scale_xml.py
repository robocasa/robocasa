import lxml.etree as ET
import argparse


def mult_str(s, factor):
    return " ".join(["{:.2f}".format(factor * float(i)) for i in s.split()])


def scale_body(body, ratio):
    for geom in body.iterfind("geom"):
        if geom.attrib["class"] != "visual":
            if "pos" in geom.attrib:
                geom.attrib["pos"] = mult_str(geom.attrib["pos"], ratio)
            if "size" in geom.attrib:
                geom.attrib["size"] = mult_str(geom.attrib["size"], ratio)

    for joint in body.iterfind("joint"):
        if "pos" in joint.attrib:
            joint.attrib["pos"] = mult_str(joint.attrib["pos"], ratio)

    for sub_body in body.iterfind("body"):
        scale_body(sub_body, ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale", type=float, required=True, help="scale value relative to base mesh"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="path of XML file to scale"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        help="(optional) path to save location. If this is not set, the original file is overwritten. Note: Repeatedly overwriting the file can eventually cause loss of precision.",
    )

    args = parser.parse_args()

    et = ET.parse(args.path)
    root = et.getroot()

    ratio = 0
    for mesh in et.iterfind("asset/mesh"):
        cur_scale = float(mesh.attrib["scale"].split()[0])
        mesh.attrib["scale"] = f"{args.scale} {args.scale} {args.scale}"
        ratio = args.scale / cur_scale

    for body in et.iterfind("worldbody/body"):
        scale_body(body, ratio)

    save_path = args.save_path if args.save_path is not None else args.path
    et.write(save_path)
