import os, sys, getopt
from os.path import join, isdir, abspath, isfile

def walk_depth(path, depth=-1):
    path = path.rstrip(os.path.sep)
    assert os.path.isdir(path)
    root_depth = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs, files
        current_depth = root.count(os.path.sep)
        if depth != -1 and (root_depth + depth <= current_depth):
            del dirs[:]

def annotateData(dataset_root, depth):

    abs_dataset_root = abspath(dataset_root)

    dataset_name = input("Dataset name: ")

    abs_dataset_data = join(abs_dataset_root, "data")

    ds = []
    for path, dirs, files in walk_depth(abs_dataset_data, depth):
        if join(abs_dataset_root, path) == abs_dataset_data:
            continue
        ds.append(path)

    ds = [d.replace(dataset_root, "") for d in ds ]

    print("Here are found directories:\n")
    print(*ds, sep='\n')

    print("\nYou'll now provide a label for each directory or exlude them from the dataset.\n")

    ls = []
    for d in ds:
        print("\nType 'exclude' to exclude directory from dataset.")
        l = input("What label should be put on samples in {}? ".format(d))
        if l == "exclude": # exclude after all labels are collected
            print("Ok, will not include {}.".format(d))
        ls.append(l)

    lds = [(d,l) for d, l in zip(ds, ls) if l != "exclude" ]

    print("")
    print(*lds, sep='\n')
    print("\n")

    confirm = input("Please confirm (yes/no): ")
    if not (confirm == "y" or confirm == "yes"):
        print("Please rerun the script and provide the correct labels.")
        exit(0)

    os.mkdir(join(abs_dataset_root, dataset_name))

    stats = {}
    annotations_file = join(abs_dataset_root, dataset_name, "annotations.csv")
    with open(annotations_file,'w+') as f:
        data = f.read()
        f.seek(0)
        f.write("# filename,label\n") # Header

        for d, l in lds:
            path = join(abs_dataset_root, d)
            files = [f for f in os.listdir(path)]

            for name in files:
                f.write("{},{}\n".format(join("..", d, name), l))
            
            if l in stats:
                stats[l] += len(files)
            else:
                stats[l] = len(files)

    print("Here are the counts for the dataset:")
    print("\n")
    for k, v in stats.items():
        print("    {}: {}".format(k,v))
    print("\nAnnotations found at {}".format(annotations_file))

def parseArgs(argv):
    def printHelp():
        print("Annotate data by creating a annotations.csv file. Provide the data root directory")
        print("and search depth to start the script. You will get be prompted for labels that")
        print("all samples in respective directory should have.\n")
        print("Args\n")
        print("     -r, --root                 Path to dataset root directory. The annotations file")
        print("                                will be created in this directory. The dir must also")
        print("                                contain a data/ dir where all samples are categories.")
        print("\n")
        print("                                Example:")
        print("                                root/")
        print("                                    data/")
        print("                                        class1/")
        print("                                            sample1")
        print("                                            sample2")
        print("                                        class2/")
        print("                                            sample1")
        print("                                            sample2")
        print("\n")
        print("     -d, --depth     (Optional) Search depth to look for label directories.")
        print("                                Default 1.")

    root = ''
    depth = -1
    try:
       opts, args = getopt.getopt(argv,"hr:d:",["root=","depth="])
    except getopt.GetoptError:
       printHelp()
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
            printHelp()
            sys.exit(0)
       elif opt in ("-r", "--root"):
            root = arg
       elif opt in ("-d", "--depth"):
            depth = int(arg)
    return root, depth if depth != -1 else 1


if __name__ == "__main__":
    root, depth = parseArgs(sys.argv[1:])
    annotateData(root, depth)
