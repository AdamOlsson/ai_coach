"""This test compares the results between converting the Human class to a numpy
array and converting a json to an numpy array. During training the training process,
jsons are generated from the Humans and saved to disc as a middle step. When
predicting new videos with a trained network, this is unessecary. Therefore, it is
essential that these two methods produce the exact same result.
"""
from os.path import join

from torchvision.transforms import Compose
import torchvision, torch
import numpy as np

from model.PoseModel.PoseModel import PoseModel

from Transformers.FactorCrop import FactorCrop
from Transformers.RTPosePreprocessing import RTPosePreprocessing
from Transformers.ToRTPoseInput import ToRTPoseInput

from util.load_config import load_config
from util.pose_estimation import poseEstimation
from util.humans_to_numpy import humansToNumpy

filename = "0xff57ad530b197149"
video_path = join("../../datasets/weightlifting/sliding_window/videos/samples/snatch", filename + ".mp4")
ndarray_path = join("../../datasets/weightlifting/ndarrays/data/snatch", filename + ".npy")

np_array_target = np.load(ndarray_path)
vframes, _, info = torchvision.io.read_video(video_path, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames

device = "cuda"
config = load_config("config.json")
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
preprocess = Compose(transformers)


vframes_proc = preprocess({"data":vframes.numpy(), "type":"video"})["data"]
poses = poseEstimation(pose_model, config, vframes_proc, device, remove_bg_objects=False)

np_array = humansToNumpy(poses)

print("Shapes:", np_array_target.shape, np_array.shape)
print("Max diff: {}".format(np.max(np.abs(np_array_target)-np.abs(np_array))))
#print(np.abs(np_array_target)-np.abs(np_array))

assert(np.equal(np_array_target, np_array).all())