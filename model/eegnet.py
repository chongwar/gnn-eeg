import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25
        
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64), bias=False),
            nn.BatchNorm2d(8)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )
        
        self.fc1 = nn.Linear((16 * (256 // 32)), num_classes)
    
    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    

if __name__ == '__main__':
    pass
