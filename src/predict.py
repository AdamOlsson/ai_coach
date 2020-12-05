# pose prediction
from model.PoseModel.PoseModel import PoseModel
from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput
from util.load_config import load_config
from util.pose_estimation import poseEstimation
from util.humans_to_numpy import humansToNumpy

from model.ExerciseModel.st_gcn.st_gcn_aaai18 import ST_GCN_18

import numpy as np
import torch, torchvision
from torchvision.transforms import Compose

import getopt, sys


# Usage:
# python predict.py -w <path to ST_GCN_18 weights> -v <path to video>
device = "cuda"

def main(weights_path, video_path, device):

    try:
        config = load_config("config.json") # hacky, if pose model weights is loaded then should config as well
    except:
        print("ERROR !!! Could not load config.json. Are you running the script from the root repository?")
        exit(1)

    # load PosePrediction network
    pose_model = PoseModel()
    pose_model.to(device)
    try:
        pose_model.load_state_dict(torch.load("model/PoseModel/weights/vgg19.pth", map_location=torch.device(device)))
    except:
        print("ERROR !!! Could not load pose model weights on path model/PoseModel/weights/vgg19.pth. Are you running the script from the root repository?")
        exit(1)
    
    # load video
    transformers = [
        FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]),
        RTPosePreprocessing(),
        ToRTPoseInput(0)]

    preprocess = Compose(transformers)
    try:
        vframes, _, info = torchvision.io.read_video(video_path, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
    except:
        print("ERROR !!! Could not load the video at {}".format(video_path))
        exit(1)
    
    print("Preprocessing sample...")
    vframes = preprocess({"data":vframes.numpy(), "type":"video"})["data"]
    print("Preprocessing done.")
    
    # extract body positions
    print("Extracting poses...")
    vframes = poseEstimation(pose_model, config, vframes, device)
    print("Extracting done.")

    del pose_model
    # body poss ndarray
    vframes = humansToNumpy(vframes)

    # predict
    layout      = config["train"]["layout"]
    strategy    = config["train"]["strategy"]
    labels      = config["labels"]

    print("Predicting...")
    graph_cfg = {"layout":layout, "strategy":strategy}
    model = ST_GCN_18(3, 3, graph_cfg, edge_importance_weighting=True, data_bn=True).to(device)

    try:
        model.load_state_dict(torch.load("../ST_GCN_18.pth", map_location=torch.device(device)))
    except:
        print("ERROR !!! Could not load model weights. Are you running the script from the root repository?")
        exit(1)

    model.eval()

    vframes = torch.from_numpy(np.expand_dims(vframes, axis=0)).float().to(device)
    print(vframes.shape)
    with torch.no_grad():
        output = model(vframes)

    print(output)
    print(labels)




def parseArgs(argv):
    try:
        opts, _ = getopt.getopt(argv, 'wv:o:', ['weights=', 'video='])
    except getopt.GetoptError:
       sys.exit(2)
    weights_path = ""
    video_path = ""
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-w", "--weights"):
            weights_path = arg
        elif opt in ("-v", "--video"):
            video_path = arg
    return weights_path, video_path 

if __name__ == "__main__":
    weights_path, video_path = parseArgs(sys.argv[1:])
    main(weights_path, video_path, device)