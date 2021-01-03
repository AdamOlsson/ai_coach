import os
from os.path import join

path_data         = "/mnt/22b72d38-1529-405a-abaf-096878d0f946/datasets/weightlifting/sliding_window/pose_predictions/data" # path relative this file
path_csv_save_loc = "/mnt/22b72d38-1529-405a-abaf-096878d0f946/datasets/weightlifting/sliding_window/pose_predictions/annotations.csv"  # path relative this file

with open(path_csv_save_loc,'w+') as f:
    data = f.read()
    f.seek(0)
    f.write("# filename,label\n") # Header
    for label in os.listdir(path_data):
        for file in os.listdir(join(path_data, label)):
            f.write("data/{}/{},{}\n".format(label, file, label)) # path, label
    f.truncate()


