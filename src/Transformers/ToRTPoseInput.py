import torch
import numpy as np

class ToRTPoseInput(object):

    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, sample):
        
        data, type = sample['data'], sample['type']
        
        if type == 'image': # Input to network is 4 dimensional
            data = np.expand_dims(data, self.dim)

        sample['data'] = torch.from_numpy(data).float()

        return sample