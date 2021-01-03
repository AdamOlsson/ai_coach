import numpy as np
import os, json
from os.path import join

"""Function to load a single sample from openpose keypoint files. It loads the json files
and creates a single numpy matrix with shape
"""

def openpose_json_body25_load_fun(sample_path):
    def to_numpy(keypoints):
        """Converts the list of keypoints to a numpy array of shape (3,300,25,1)"""
        C = 3; T = len(keypoints); V = 25; M = 1
        keypoints = np.reshape(keypoints, (C, T, V, M))

        if T > 300:
            keypoints = keypoints[:,:300,:,:] # cut length of sample to 300 frames
        if T < 300:
            # replay the earliest frames to fill 300 frames
            keypoints_300 = np.empty((C, 300, V, 1))
            for k in range(300):
                keypoints_300[:,k,:,:] = keypoints[:,k%T,:,:]

            keypoints = keypoints_300
        
        return keypoints


    """Each sample_path is a directory of keypoint files. Each keypoint files contain
    keypoints for a single frame in a video.
    """
    keypoint_files = [f for f in os.listdir(sample_path) if f.endswith("keypoints.json")]
    keypoints = [None] * len(keypoint_files)
    for keypoint_file in keypoint_files:
        with open(join(sample_path, keypoint_file)) as f:
            data = json.load(f)
        
        keypoints_people = data["people"] # list
        assert len(keypoints_people) == 1 # We are only interested in one person

        keypoints_person = keypoints_people[0]["pose_keypoints_2d"]

        # extract frame idx
        frame_idx_str_start = len(keypoint_file) - keypoint_file[::-1].find("_", len("_keypoints.json"))
        frame_idx = int(keypoint_file[frame_idx_str_start: -len("_keypoints.json")])

        keypoints[frame_idx] = keypoints_person
    
    return to_numpy(keypoints)