import torch 
import torch.nn as nn 
import torchvision.models as models
import torch.nn.functional as F
import gc

class TridentResNet2(nn.Module):
    def __init__(self, pretrained):
        super(TridentResNet2, self).__init__()
        if pretrained:
            self.resnet_1 = models.resnet18(pretrained=True)
            self.resnet_2 = models.resnet18(pretrained=True)
            self.resnet_3 = models.resnet18(pretrained=True)
        else:
            self.resnet_1 = models.resnet18(pretrained=False)
            self.resnet_2 = models.resnet18(pretrained=False)
            self.resnet_3 = models.resnet18(pretrained=False)
        """
        self.resnet_1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_3.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        """

        self.resnet_1 = nn.Sequential(*list(self.resnet_1.children())[:-2])
        self.resnet_2 = nn.Sequential(*list(self.resnet_2.children())[:-2])
        self.resnet_3 = nn.Sequential(*list(self.resnet_3.children())[:-2])
        
        self.last_block = nn.Sequential(
            nn.Conv2d(512,64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x1 = self.resnet_1(x[:,:,:64,:])
        x2 = self.resnet_2(x[:,:,64:128,:])
        x3 = self.resnet_3(x[:,:,128:,:])
        
        x = torch.cat([x1,x2,x3], axis=2)
        x = self.last_block(x).squeeze()
        return x
        