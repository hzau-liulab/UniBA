import argparse
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np


def parse_xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    interaction_tags = {
        "hydrophobic_interactions": 0,
        "hydrogen_bonds": 1,
        "salt_bridges": 2,
        "pi_stacks": 3,
        "pi_cation_interactions": 4,
    }

    plip_dict = {}
    for bs in root.findall(".//bindingsite"):
        interactions = bs.find("interactions")
        if interactions is None:
            continue

        for tag, idx in interaction_tags.items():
            block = interactions.find(tag)
            if block is None:
                continue

            for item in block:
                try:
                    c1 = item.find("reschain").text.strip()
                    r1 = int(item.find("resnr").text)
                    c2 = item.find("reschain_lig").text.strip()
                    r2 = int(item.find("resnr_lig").text)

                    key = (c1, r1, c2, r2)
                    if key not in plip_dict:
                        plip_dict[key] = [0, 0, 0, 0, 0]
                    plip_dict[key][idx] = 1
                except Exception:
                    continue

    return plip_dict


def build_all_plip_dict(pair_list, plip_root, save_path):
    all_data = {}

    for i, pair in enumerate(pair_list):
        pair_dir = os.path.join(plip_root, str(pair))
        xml_candidates = [
            os.path.join(pair_dir, f"{pair}_report.xml"),
            os.path.join(pair_dir, "report.xml"),
        ]
        xml_file = next((path for path in xml_candidates if os.path.exists(path)), None)
        if xml_file is None:
            print(f"[missing PLIP XML] {pair_dir}")
            continue
        print(i, pair)
        plip_dict = parse_xml_to_dict(xml_file)
        all_data[str(pair)] = plip_dict

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"[Done] Saved {len(all_data)} PLIP entries to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a pickle of PLIP interaction features from per-pair XML reports."
    )
    parser.add_argument("--pair-list", required=True, help="Text file of pair ids")
    parser.add_argument("--plip-root", required=True, help="Directory containing per-pair PLIP outputs")
    parser.add_argument("--output", required=True, help="Output .pkl file")
    return parser.parse_args()


def main():
    args = parse_args()
    pair_list = np.loadtxt(args.pair_list, dtype=str)
    if pair_list.ndim == 0:
        pair_list = np.array([str(pair_list)])
    build_all_plip_dict(pair_list, args.plip_root, args.output)


if __name__ == "__main__":
    main()
