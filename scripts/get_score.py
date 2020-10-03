# LAZY IMPORT
import os
import sys
cur_dir = os.path.dirname(__file__)
master_dir = os.path.join("..", cur_dir)
sys.path.append(master_dir)
##############################
import argparse
import tqdm

from pixel_privacy.models.biqa_model import BIQAModel



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path", 
        "-w",
        help="Path to BIQA model weight",
    )

    parser.add_argument(
        "--flist_path", 
        "-f",
        help="Path to csv folder contains list image",
    )
    parser.add_argument(
        "--input_folder", 
        "-i",
        help="Path to input folder",
    )

    parser.add_argument(
        "--output_csv", 
        "-o",
        help="Path to output csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = BIQAModel(args.weight_path, pretrained=None)
    with open(args.flist_path) as fin:
        fname_list = [line.strip() for line in fin.readlines()]

    fout = open(args.output_csv, "w")
    fout.write("file_name,score\n")
    for i, fname in enumerate(tqdm.tqdm(fname_list)):
        true_name = fname[:len(fname)-3] + "png"
        fpath = os.path.join(args.input_folder, true_name)
        score = model.process(fpath)
        fout.write(f"{fname},{score}\n")

    fout.close()
    print("Done")


if __name__ == "__main__":
    main()
