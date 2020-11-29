import os

root              = "exercises"
path_data         = "../datasets/weightlifting/videos/{}/".format(root) # path relative this file
path_csv_save_loc = "../datasets/weightlifting/videos/annotations.csv"  # path relative this file

with open(path_csv_save_loc,'w+') as f:
    data = f.read()
    f.seek(0)
    f.write("# filename,label\n") # Header
    for dir in os.listdir(path_data):
        for file in os.listdir(path_data + dir):
            f.write("{}/{}/{},{}\n".format(root, dir, file, dir)) # path, label
    f.truncate()


