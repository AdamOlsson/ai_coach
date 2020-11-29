import pandas as pd
from os.path import exists, join, splitext, isfile
from os import makedirs, mkdir, listdir
from shutil import rmtree
import torchvision, torch

"""
samples/snatch/00/00.mp4,snatch
samples/snatch/00/01.mp4,snatch
"""
def main(input_dir, output_dir):
    annotations_path = join(input_dir, "annotations.csv")
    annotations = pd.read_csv(annotations_path)

    filenames = annotations.iloc[:,0]
    sample_parent_dirs = [join(input_dir, join(*f.split("/")[0:-1])) for f in filenames]

    sample_parent_dirs = list(set(sample_parent_dirs)) # remove duplicates, note that order is removed now

    for i, d in enumerate(sample_parent_dirs):
        files = sorted([f for f in listdir(d) if isfile(join(d, f))], reverse=True)
        vids = []
        for f in files:
            vid_name = join(d,f)
            print(vid_name)
            vframes, _, info = torchvision.io.read_video(vid_name, pts_unit="sec")
            vids.append(vframes)

        full_video = torch.cat(vids, 0)

        # extract label from path
        label = d.split("/")[-2]

        # write video and fill in annotations file
        fps = 30
        filename =  str(i) + ".mp4"
        save_path = join(output_dir, "data", label, filename)
        torchvision.io.write_video(save_path, full_video, fps)

        with open(join(output_dir, "annotations.csv"), "a") as f:
            f.write("data/{},{}\n".format(join(label, filename), label))

        print("Merging... {} out of {}".format(i, len(sample_parent_dirs)))
    
    print("Done!")



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


if __name__ == "__main__":
    input_dir = "../datasets/weightlifting/sliding_window/slided"
    output_dir = "../datasets/weightlifting/sliding_window"
    output_dir = setup(input_dir, output_dir, "merged", "filename,label")
    main(input_dir, output_dir)

