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
        """
        self.resnet_1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_3.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        """

        self.resnet_1.fc = nn.Linear(512, 256)
        self.resnet_2.fc = nn.Linear(512, 256)
        self.resnet_3.fc = nn.Linear(512, 256)

        """
        self.resnet_1 = nn.Sequential(*list(self.resnet_1.children()))
        self.resnet_2 = nn.Sequential(*list(self.resnet_2.children()))
        self.resnet_3 = nn.Sequential(*list(self.resnet_3.children()))
        """ 
        
        self.last_block = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.ReLU(), 
            nn.Linear(768, 256),
            nn.ReLU(), 
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Linear(64, 10),
            nn.ReLU(), 
        )
        
        
    def forward(self, x):
        #x = x.unsqueeze(1).transpose(2,3)
        x1 = self.resnet_1(x[:,:,:64,:])
        x2 = self.resnet_2(x[:,:,64:128,:])
        x3 = self.resnet_3(x[:,:,128:,:])
        """ 
        x1 = self.fc_1(x1)
        x2 = self.fc_2(x2)
        x3 = self.fc_3(x3)
        """
        x = torch.cat([x1,x2,x3], axis=1)
        x = self.last_block(x)
        return x
        
