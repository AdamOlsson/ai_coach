# pose prediction
from PosePrediction.model.PoseModel import PoseModel
from PosePrediction.Transformers.FactorCrop import FactorCrop
from PosePrediction.Transformers.RTPosePreprocessing import RTPosePreprocessing
from PosePrediction.Transformers.ToRTPoseInput import ToRTPoseInput
from PosePrediction.util.load_config import load_config
from PosePrediction.pose_estimation import poseEstimation

import numpy as np
import torch, torchvision
from torchvision.transforms import Compose

import getopt, sys


# Usage:
# python predict.py -w <path to ST_GCN_18 weights> -v <path to video>
device = "cuda"


def main(weights_path, video_path, device):

    try:
        config = load_config("PosePrediction/config.json") # hacky, if pose model weights is loaded then should config as well
    except:
        print("ERROR !!! Could not load config PosePrediction/config.json. Are you running the script from the root repository?")
        exit(1)

    # load PosePrediction network
    pose_model = PoseModel()
    pose_model.to(device)
    try:
        pose_model.load_state_dict(torch.load("PosePrediction/model/weights/vgg19.pth", map_location=torch.device(device)))
    except:
        print("ERROR !!! Could not load pose model weights on path PosePrediction/model/weights/vgg19.pth. Are you running the script from the root repository?")
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
    
    vframes = preprocess({"data":vframes.numpy(), "type":"video"})["data"]

    # extract body positions
    
    # body poss ndarray
    # predict

    pass


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