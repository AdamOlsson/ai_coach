# native
import sys, getopt, json
from os.path import join, basename, splitext

# misc
import pandas as pd # easy load of csv
from util.setup_directories import setup
from paf.util import load_humans, save_humans
from util.remove_bg_humans import remove_bg_humans

def load_partial_jsons(path, no_partial_graphs):
    dicts = [None]*(no_partial_graphs+1)

    for i in range(no_partial_graphs+1):
        path_partial_graphs = join(path, str(i) + '.json')
        with open(path_partial_graphs) as f:
            dicts[i] = json.load(f)
    
    return dicts

def merge_dicts(dicts):

    new_frames = []
    frame_count = 0
    for d in dicts:
        f = d["frames"]
        frame_count += len(f)
        new_frames.extend(d["frames"])

    # assert correct ordering
    for i in range(1, len(new_frames)):
        prev_frame = new_frames[i-1]
        current_frame = new_frames[i]
        assert(prev_frame["frame_id"] < current_frame["frame_id"] or current_frame["frame_id"] == 0)
    
    # assign new frame id
    for i in range(len(new_frames)):
        new_frames[i]["frame_id"] = i

    metadata = dicts[0]["metadata"]
    metadata["filename"] = basename(metadata["filename"])
    del metadata["subpart"] # only used in this script

    new_dict = {}
    new_dict["frames"] = new_frames
    new_dict["metadata"] = metadata

    return new_dict 


def main(input_dir, output_dir):
    annotations_path = join(input_dir, "annotations.csv")
    annotations = pd.read_csv(annotations_path)
    
    annotations_out = join(output_dir, "annotations.csv")

    paths = annotations.iloc[:,0]
    labels = annotations.iloc[:,1]
    no_partial_graphs = annotations.iloc[:,2]
    for p, l, no in zip(paths, labels, no_partial_graphs):
        dicts = load_partial_jsons(join(input_dir, p), no)
        merged_dicts = merge_dicts(dicts)

        merged_dicts["frames"] = remove_bg_humans(merged_dicts["frames"])

        name, _ = splitext(basename(p))
        filename = join(output_dir, "data", l, name + ".json")
        print("Merged {}".format(filename))

        with open(filename, 'w') as f:
            f.write(json.dumps(merged_dicts, indent=4, sort_keys=True))

        annotations_filename = join("data", l, name + ".json")
        with open(annotations_out, "a") as f:
            f.write("{},{},{}\n".format(annotations_filename, l, len(merged_dicts["frames"])))


    print("\nOutput can be found at {}".format(output_dir))

def parse_args(argv):
    try:
        opts, _ = getopt.getopt(argv, 'hi:o:', ['input_dir=', 'output_dir='])
    except getopt.GetoptError:
       sys.exit(2)
    input_dir = ""
    output_dir = ""
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-i", "--input"):
            input_dir = arg
        elif opt in ("-o", "--output"):
            output_dir = arg
    return input_dir, output_dir 

if __name__ == "__main__":
    input_dir, output_dir = parse_args(sys.argv[1:])
    output_dir = setup(input_dir, output_dir, "graphs", "filename,label,no_frames")
    main(input_dir, output_dir)