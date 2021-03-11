import torch 
import torch.nn as nn 
import torchvision.models as models
import torch.nn.functional as F
import gc
class TridentResNet(nn.Module):
    def __init__(self, pretrained):
        super(TridentResNet, self).__init__()
        if pretrained:
            self.resnet_1 = models.resnet18(pretrained=True)
            self.resnet_2 = models.resnet18(pretrained=True)
            self.resnet_3 = models.resnet18(pretrained=True)
        else:
            self.resnet_1 = models.resnet18(pretrained=False)
            self.resnet_2 = models.resnet18(pretrained=False)
            self.resnet_3 = models.resnet18(pretrained=False)
        self.resnet_1 = nn.Sequential(*list(self.resnet_1.children())[:-3])
        self.resnet_2 = nn.Sequential(*list(self.resnet_2.children())[:-3])
        self.resnet_3 = nn.Sequential(*list(self.resnet_3.children())[:-3])
        self.fc_1 = nn.Linear(4,64)
        self.fc_2 = nn.Linear(4,64)
        self.fc_3 = nn.Linear(8,128)
        
        self.last_block = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.Conv2d(768,10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        
    def forward(self, x):
        x = x.transpose(2,3)
        x1 = self.resnet_1(x[:,:,:,:64])
        x2 = self.resnet_2(x[:,:,:,64:128])
        x3 = self.resnet_3(x[:,:,:,128:])
        
        x1 = self.fc_1(x1)
        x2 = self.fc_2(x2)
        x3 = self.fc_3(x3)
        x = torch.cat([x1,x2,x3], axis=3)
        x = self.last_block(x).squeeze(2).squeeze(2)
        return x
        
