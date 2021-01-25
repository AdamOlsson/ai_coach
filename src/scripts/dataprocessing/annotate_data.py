import os
from os.path import join, isdir, abspath, isfile

def annotateData(dataset_root):

    abs_dataset_root = abspath(dataset_root)

    dataset_name = input("Dataset name: ")

    ds = []
    for path, dirs, files in os.walk(join(abs_dataset_root, "data")):
        if len(files) != 0:
            ds.append(path)

    ds = [d.replace(dataset_root, "") for d in ds ]

    print("Here are found directories:\n")
    print(*ds, sep='\n')

    print("\nYou'll now provide a label for each directory.\n")

    ls = []
    for d in ds:
        l = input("What label should be put on samples in {}? ".format(d))
        ls.append(l)


    print("")
    print(*zip(ds, ls), sep='\n')
    print("\n")

    confirm = input("Please confirm (yes/no): ")
    if not (confirm == "y" or confirm == "yes"):
        print("Please rerun the script and provide the correct labels.")
        exit(0)

    os.mkdir(join(abs_dataset_root, dataset_name))

    annotations_file = join(abs_dataset_root, dataset_name, "annotations.csv")
    with open(annotations_file,'w+') as f:
        data = f.read()
        f.seek(0)
        f.write("# filename,label\n") # Header

        for (d, l) in zip(ds, ls):
            path = join(abs_dataset_root, d)
            files = [f for f in os.listdir(path) if ".mp4" in f]

            for name in files:
                f.write("{},{}\n".format(join("..", d, name), l))

    print("Annotations found at {}".format(annotations_file))
    



if __name__ == "__main__":
    annotateData("/media/adam/G/datasets/weightlifting/telegram2/videos/")
