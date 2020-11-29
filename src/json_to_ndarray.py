"""
This script converts the json files with graph data from PosePrediction to ndarrays. These
ndarrays serve as input to ST-GCN. The ndarray data is (C, T, V, M) where

    C = 3       (['x', 'y', 'score'])
    T = 300     (no frames)
    V = 18      (no of nodes in human skeleton)
    M = 1       (no humans in each frame)

according to https://arxiv.org/pdf/1801.07455.pdf


The output directory receives the following format:

<output_dir>/
    annotations.csv
    data/
        <label1>/
            <file>.npy
            ...
        <label2>/
            <file>.npy
            ...
"""

# local
from PosePrediction.util.setup_directories import setup

# other
import numpy as np
import pandas as pd

# native
from os.path import join, basename, splitext
import sys, getopt, json

def parse_args(argv):
    try:
        opts, _ = getopt.getopt(argv, 'hi:o:', ['input_dir=', 'output_dir='])
    except getopt.GetoptError:
       print('json_to_ndarray.py --input_dir <inputdir> --out_dir <outdir>')
       sys.exit(2)

    input_dir = ""
    output_dir = ""
    for opt, arg in opts:
        if opt == '-h':
            print('json_to_ndarray.py -i <inputdir> -o <outdir>')
            sys.exit()
        elif opt in ("-i", "--input_dir"):
            input_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
    return input_dir, output_dir 

def main(input_dir, output_dir):
    
    C = 3
    T = 300
    V = 18
    M = 1
    data_numpy = np.zeros((C, T, V, M))

    annotations_in = pd.read_csv(join(input_dir, "annotations.csv"))
    filenames   = annotations_in.iloc[:,0]
    labels      = annotations_in.iloc[:,1]

    annotations_out = join(output_dir, "annotations.csv")

    input_filenames = [join(input_dir, f) for f in filenames]

    for i, name in enumerate(input_filenames):
        print("{:6d} ::: Processing {}".format(i, name))
        with open(name) as f:
            data = json.load(f)
 
        frames = data["frames"]
 
        for t in range(T):
            if t >= T: # Clip videos to T frames
                break
            frame = frames[t%len(frames)] # replay frames for padding
            bodies = frame["bodies"]
            frame_id = frame["frame_id"]

            # If no body found in frame, skip
            if len(bodies) == 0:
                continue                

            body_parts = bodies[0]["body_parts"]
            for part_id, part_data in body_parts.items():
                data_numpy[0, frame_id, int(part_id), 0] = part_data["x"]
                data_numpy[1, frame_id, int(part_id), 0] = part_data["y"]
                data_numpy[2, frame_id, int(part_id), 0] = part_data["score"]

        fname, _      = splitext(basename(name))
        fname = fname + ".npy"
        dataset_name = join("data", labels[i], fname)
        savepath     = join(output_dir, dataset_name)

        np.save(savepath, data_numpy)

        with open(annotations_out, "a") as f:
            f.write("{},{},{},{},{},{}\n".format(dataset_name, labels[i], C,T,V,M))


    return data_numpy


if __name__ == "__main__":
    input_dir, output_dir = parse_args(sys.argv[1:])
    output_dir = setup(input_dir, output_dir, "ndarrays", "filename,label,C,T,V,M")
    data = main(input_dir, output_dir)