from os.path import exists, join, basename, splitext
from shutil import rmtree
from os import makedirs, mkdir

import pandas as pd # easy load of csv


def setup(input_dir, output_dir, root_dir_name, csv_header):
    ## Setup structure of the output directory.
    annotations_path = join(input_dir, "annotations.csv")
    annotations = pd.read_csv(annotations_path)
    
    labels = annotations.iloc[:,1]
    unique_labels = list(set(labels))

    data_out_root = join(output_dir, root_dir_name)

    # delete data, start from clean slate
    if exists(data_out_root):
        rmtree(data_out_root)
    
    data_out = join(data_out_root, "data")
    makedirs(data_out) # recursive create of root and data dir

    for label in unique_labels:
        path = join(data_out, label)
        mkdir(path)

    annotations_out = join(data_out_root, "annotations.csv") 

    with open(annotations_out,'w+') as f:
        f.write("# {}\n".format(csv_header)) # Header

    return data_out_root