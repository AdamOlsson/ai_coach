from model.ExerciseModel.st_gcn.st_gcn_aaai18 import ST_GCN_18
from util.load_config import load_config

import numpy as np
import torch

filename = "../../datasets/weightlifting/testing/ndarrays/data/snatch/sn3.npy"
device = "cuda"

video = np.load(filename)
video = video[np.newaxis, :] # add batch dim to data
video = torch.tensor(video, dtype=torch.float32)

config = load_config("config.json")

layout      = config["train"]["layout"]
strategy    = config["train"]["strategy"]
labels      = config["labels"]

graph_cfg = {"layout":layout, "strategy":strategy}
no_classes = 3
model = ST_GCN_18(3, no_classes, graph_cfg, edge_importance_weighting=True, data_bn=True).to(device)
model.load_state_dict(torch.load("../ST_GCN_18.pth", map_location=torch.device(device)))



model.eval()

with torch.no_grad():
    output = model(video.to(device))
output = output[0].cpu() # batch 0

print("{:12}: {}\n{:12}: {}\n{:12}: {}\n".format("Snatch", output[labels["snatch"]], "Frontsquat",output[labels["frontsquat"]], "Backsquat", output[labels["backsquat"]]))
