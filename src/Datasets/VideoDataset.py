import pandas as pd # easy load of csv
import cv2
import numpy as np

from os.path import join

import torch, torchvision
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, path_csv, path_root, transform=None, load_copy=False, frame_skip=1):
        self._path_csv  = path_csv
        self._path_root = path_root
        self.annotations = pd.read_csv(path_csv)
        self.transform = transform
        self.load_copy = load_copy
        self.frame_skip = frame_skip


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vid_name = join(self._path_root,self.annotations.iloc[idx,0])

        vframes, _, info = torchvision.io.read_video(vid_name, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
        label = self.annotations.iloc[idx,1]

        vframes = np.flip(vframes.numpy(), axis=3)

        if self.frame_skip != 0:
            no_frames = vframes.shape[0]
            selected_frames = np.linspace(0, no_frames-1, num=int(no_frames/self.frame_skip), dtype=np.int)
            vframes = vframes[selected_frames]

        sample = {'data':vframes, 'label':label, 'name':vid_name, 'type':'video', "properties":info}

        if self.load_copy:
            sample['copy'] = np.copy(vframes)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
