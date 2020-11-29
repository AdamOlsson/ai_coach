import torch
import torch.nn as nn
import torch.nn.functional as F
from model.FeatureExtractors.VGG import VGG19FeatureExtractor

class Block5(nn.Module):
    def __init__(self, channels_out):
        super(Block5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) # {'conv5_1_CPM_L1': [128, 128, 3, 1, 1]}
        self.ReLU1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) # {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]}
        self.ReLU2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) # {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]}
        self.ReLU3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0) # {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]}
        self.ReLU4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=channels_out, kernel_size=1, stride=1, padding=0)  # {'conv5_5_CPM_L1': [512, 38/19, 1, 1, 0]}

    def forward(self, x):
        y = self.ReLU1(self.conv1(x))
        y = self.ReLU2(self.conv2(y))
        y = self.ReLU3(self.conv3(y))
        y = self.ReLU4(self.conv4(y))
        y = self.conv5(y)      
        return y

class Block7(nn.Module):
    def __init__(self, channels_out):
        super(Block7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=185, out_channels=128, kernel_size=7, stride=1, padding=3) # {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]}
        self.ReLU1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3) # {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]}
        self.ReLU2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3) # {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]}
        self.ReLU3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3) # {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]}
        self.ReLU4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3) # {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]}
        self.ReLU5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0) # {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]}
        self.ReLU6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=channels_out, kernel_size=1, stride=1, padding=0) # {'Mconv7_stage%d_L1' % i: [128, 38/19, 1, 1, 0]}

    def forward(self, x):
        y = self.ReLU1(self.conv1(x))            
        y = self.ReLU2(self.conv2(y))        
        y = self.ReLU3(self.conv3(y))        
        y = self.ReLU4(self.conv4(y))        
        y = self.ReLU5(self.conv5(y))        
        y = self.ReLU6(self.conv6(y))        
        y = self.conv7(y)
        return y      


class PoseModel(nn.Module):
    def __init__(self):
        super(PoseModel, self).__init__()

        self.preprocessing = VGG19FeatureExtractor()

        self.stage1_branch1 = Block5(channels_out=38)
        self.stage2_branch1 = Block7(channels_out=38)
        self.stage3_branch1 = Block7(channels_out=38)
        self.stage4_branch1 = Block7(channels_out=38)
        self.stage5_branch1 = Block7(channels_out=38)
        self.stage6_branch1 = Block7(channels_out=38)

        self.stage1_branch2 = Block5(channels_out=19)
        self.stage2_branch2 = Block7(channels_out=19)
        self.stage3_branch2 = Block7(channels_out=19)
        self.stage4_branch2 = Block7(channels_out=19)
        self.stage5_branch2 = Block7(channels_out=19)
        self.stage6_branch2 = Block7(channels_out=19)


    def forward(self, x):

        stage_losses = []
        out_pre = self.preprocessing(x)

        out_stage1_branch1 = self.stage1_branch1(out_pre)
        out_stage1_branch2 = self.stage1_branch2(out_pre)
        out_stage1 = torch.cat([out_stage1_branch1, out_stage1_branch2, out_pre], 1)
        stage_losses.append((out_stage1_branch1, out_stage1_branch2))

        out_stage2_branch1 = self.stage2_branch1(out_stage1)
        out_stage2_branch2 = self.stage2_branch2(out_stage1)
        out_stage2 = torch.cat([out_stage2_branch1, out_stage2_branch2, out_pre], 1)
        stage_losses.append((out_stage2_branch1, out_stage2_branch2))

        out_stage3_branch1 = self.stage3_branch1(out_stage2)
        out_stage3_branch2 = self.stage3_branch2(out_stage2)
        out_stage3 = torch.cat([out_stage3_branch1, out_stage3_branch2, out_pre], 1)
        stage_losses.append((out_stage3_branch1, out_stage3_branch2))

        out_stage4_branch1 = self.stage4_branch1(out_stage3)
        out_stage4_branch2 = self.stage4_branch2(out_stage3)
        out_stage4 = torch.cat([out_stage4_branch1, out_stage4_branch2, out_pre], 1)
        stage_losses.append((out_stage4_branch1, out_stage4_branch2))

        out_stage5_branch1 = self.stage5_branch1(out_stage4)
        out_stage5_branch2 = self.stage5_branch2(out_stage4)
        out_stage5 = torch.cat([out_stage5_branch1, out_stage5_branch2, out_pre], 1)
        stage_losses.append((out_stage5_branch1, out_stage5_branch2))

        out_stage6_branch1 = self.stage6_branch1(out_stage5)
        out_stage6_branch2 = self.stage6_branch2(out_stage5)
        stage_losses.append((out_stage6_branch1, out_stage6_branch2))

        return (out_stage6_branch1, out_stage6_branch2), stage_losses
