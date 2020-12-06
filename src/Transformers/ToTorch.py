import torch

class ToTorch(object):
    """Converts numpy color axis (H x W x C) to
    torch color axis (C X H X W)"""

    def __init__(self):
        pass

    def __call__(self, sample):
        
        data = sample['data']

        sample['data'] = data.transpose((2, 0, 1))
        return sample