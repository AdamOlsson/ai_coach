import numpy as np

class ExpandDim(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        
        data, label = sample['data'], sample['label']

        return {'data':np.expand_dims(data, self.dim), 'label':label}