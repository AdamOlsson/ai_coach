import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        
        data = sample['data']

        # swap color axis because
        # numpy data: H x W x C
        # torch data: C X H X W
        sample['data'] = data.transpose((2, 0, 1))
        return sample