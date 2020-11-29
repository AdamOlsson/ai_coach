import numpy as np

class RTPosePreprocessing(object):

    def __init__(self):
        pass

    def __call__(self, sample):

        data = sample['data']
        data = data.astype(np.float32)
        data = data / 256 - 0.5

        type = sample['type']
        if type == 'image':     # Assuming 3 dims [H, W, C]
            data = data.transpose((2, 0, 1)).astype(np.float32)
        elif type == 'video':   # Assuming 4 dims [T, H, W, C]
            data = data.transpose((0, 3, 1, 2)).astype(np.float32)
        else:                   # Not yet implemented
            raise NotImplementedError("Data of type '{}' has yet not been implemented with this transformer.".format(type))
        
        sample['data'] = data
        return sample