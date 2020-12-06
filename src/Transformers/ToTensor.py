import torch

class ToTensor(object):
    """Convert ndarrays to Tensors."""

    def __init__(self, dtype=torch.float32, requires_grad=False, device="cpu"):
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.device = device

    def __call__(self, sample):
        
        data = sample['data']

        sample['data'] = torch.tensor(data, dtype=self.dtype, requires_grad=self.requires_grad).to(self.device)
        return sample