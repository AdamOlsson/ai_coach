import torch.nn as nn
import torch.nn.functional as F

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()

        self.conv1  = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)     # {'conv1_1': [3, 64, 3, 1, 1]}
        self.ReLU1  = nn.ReLU(inplace=True)
        self.conv2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)    # {'conv1_2': [64, 64, 3, 1, 1]}
        self.ReLU2  = nn.ReLU(inplace=True)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                  # {'pool1_stage1': [2, 2, 0]}
        
        self.conv3  = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)   # {'conv2_1': [64, 128, 3, 1, 1]}
        self.ReLU3  = nn.ReLU(inplace=True)
        self.conv4  = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # {'conv2_2': [128, 128, 3, 1, 1]}
        self.ReLU4  = nn.ReLU(inplace=True)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                  # {'pool2_stage1': [2, 2, 0]}

        self.conv5  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # {'conv3_1': [128, 256, 3, 1, 1]}
        self.ReLU5  = nn.ReLU(inplace=True)
        self.conv6  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # {'conv3_2': [256, 256, 3, 1, 1]}
        self.ReLU6  = nn.ReLU(inplace=True)
        self.conv7  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # {'conv3_3': [256, 256, 3, 1, 1]}
        self.ReLU7  = nn.ReLU(inplace=True)
        self.conv8  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # {'conv3_4': [256, 256, 3, 1, 1]}
        self.ReLU8  = nn.ReLU(inplace=True)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                  # {'pool3_stage1': [2, 2, 0]}

        self.conv9  = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # {'conv3_3': [256, 512, 3, 1, 1]}
        self.ReLU9  = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # {'conv3_4': [512, 512, 3, 1, 1]}
        self.ReLU10 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1) # {'conv3_3': [512, 256, 3, 1, 1]}
        self.ReLU11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1) # {'conv3_4': [256, 128, 3, 1, 1]}
        self.ReLU12 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.ReLU1(self.conv1(x))
        y = self.ReLU2(self.conv2(y))
        y = self.pool1(y)
        y = self.ReLU3(self.conv3(y))
        y = self.ReLU4(self.conv4(y))
        y = self.pool2(y)
        y = self.ReLU5(self.conv5(y))
        y = self.ReLU6(self.conv6(y))
        y = self.ReLU7(self.conv7(y))
        y = self.ReLU8(self.conv8(y))
        y = self.pool3(y)
        y = self.ReLU9(self.conv9(y))
        y = self.ReLU10(self.conv10(y))
        y = self.ReLU11(self.conv11(y))
        y = self.ReLU12(self.conv12(y))
        return y