from util.load_config import load_config
from model.ExerciseModel.st_gcn.st_gcn_aaai18 import ST_GCN_18
from load_functions.openpose_json_body25_load_fun import openpose_json_body25_load_fun

import numpy as np
import torch, torchvision

import getopt, sys


# Usage:
# python predict.py -w <path to ST_GCN_18 weights> -v <path to video>
device = "cuda"

def main(weights_path, json_path, device):
    def rank_and_print_outputs(output, labels):
        len_dataset_labels = max(labels.values()) +1
        labels_list_ordered = [None] * len_dataset_labels

        for k, v in labels.items():
            labels_list_ordered[v] = k

        # sort output after score
        rankings = [(label, score) for label, score in sorted(zip(labels_list_ordered, output), key=lambda pair: pair[1], reverse=True)]
        print("\nResults:")
        for i, (l, s) in enumerate(rankings):
            print("{}.  {:15} with score {}".format(i+1, l, s))
        print("\n")

    config = load_config("config.json") # hacky, if pose model weights is loaded then should config as well

    # predict
    layout      = config["train"]["layout"]
    strategy    = config["train"]["strategy"]
    labels      = config["labels"]

    data = openpose_json_body25_load_fun(json_path)
    data = np.expand_dims(data, axis=0) # expand batch axis
    data = torch.tensor(data, dtype=torch.float32, requires_grad=False).to(device)

    print("Predicting...")
    graph_cfg = {"layout":layout, "strategy":strategy}
    len_dataset_labels = max(labels.values()) +1
    model = ST_GCN_18(3, len_dataset_labels, graph_cfg, edge_importance_weighting=True, data_bn=True).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()

    with torch.no_grad():
        output = model(data)

    rank_and_print_outputs(output[0].tolist(), labels)
    print(*zip(labels, output[0].tolist()))


def parseArgs(argv):
    weights = ''
    json = ''
    try:
       opts, args = getopt.getopt(argv,"hw:j:",["weights=","json="])
    except getopt.GetoptError:
       print('test.py -w <weights> -j <json>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('test.py -w <weights> -j <json>')
          sys.exit()
       elif opt in ("-w", "--weights"):
          weights = arg
       elif opt in ("-j", "--json"):
          json = arg
    return weights, json

if __name__ == "__main__":
    weights_path, json_path = parseArgs(sys.argv[1:])
    main(weights_path, json_path, device)