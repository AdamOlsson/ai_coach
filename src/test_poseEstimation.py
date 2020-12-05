"""This test evaulates if the poseEstimation function produces the correct result
by extracting poses and rendering them on a video. Verification works by visually
inspecting the rendered video and by saving the pose to a json and comparing to
the json in the training data.
"""

from torchvision.transforms import Compose
import torchvision, torch
import numpy as np

from model.PoseModel.PoseModel import PoseModel

from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

from util.load_config import load_config
from util.pose_estimation import poseEstimation
from paf.common import draw_humans

device = "cuda"
config = load_config("config.json")
video_path = "../../datasets/weightlifting/testing/samples/bs3.mp4"
pose_model_weights = "model/PoseModel/weights/vgg19.pth"

pose_model = PoseModel()
pose_model.to(device)
pose_model.load_state_dict(torch.load(pose_model_weights, map_location=torch.device(device)))
pose_model.eval()

transformers = [
    FactorCrop(config["model"]["downsample"], dest_size=config["dataset"]["image_size"]),
    RTPosePreprocessing(),
    ToRTPoseInput(0)]

# load video
vframes, _, info = torchvision.io.read_video(video_path, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
preprocess = Compose(transformers)


vframes_proc = preprocess({"data":vframes.numpy(), "type":"video"})["data"]
poses = poseEstimation(pose_model, config, vframes_proc, device, remove_bg_objects=False)
del vframes_proc

vframes = vframes.numpy()
for i, frame in enumerate(vframes):
    vframes[i] = draw_humans(np.float32(frame), poses[i])

filename = "../results/test_pose_estimation.mp4"
torchvision.io.write_video(filename, torch.from_numpy(vframes), 30)

print("Video written to {}".format(filename))

# save pose as json

# load pose again

# load pose from training data

# compare, they should be roughly equal
