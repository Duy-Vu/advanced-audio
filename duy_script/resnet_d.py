# Source: https://github.com/MihawkHu/DCASE2020_task1/blob/2384da9d357bc6be9648701a0ae0c4b4be285e36/task1a/10class/resnet/resnet.py
from torch import Tensor, cat, reshape, rand
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU,\
                        Dropout2d, Linear, AdaptiveAvgPool2d
from torchvision.models import resnet18
from torch.optim import Adam

__author__ = 'Duy Vu'

class ResNet_d(Module):
    def __init__(self, num_classes=10) -> None:
        super(ResNet_d, self).__init__()
        
        # Get architecture of ResNet-18 but without the last pooling layer and FC layer
        self.res_net_1 = resnet18(pretrained=False) 
        self.res_net_2 = resnet18(pretrained=False)
        
        self.res_net_1 = Sequential(*(list(self.res_net_1.children())[:-2]))
        self.res_net_2 = Sequential(*(list(self.res_net_2.children())[:-2]))

        self.conv_out_1 = Sequential(
            Conv2d(in_channels=512,
                   out_channels=64,
                   kernel_size=(1, 1),
                   stride=(1, 1)),
            BatchNorm2d(num_features=64),
            ReLU()
        )
        
        self.conv_out_2 = Sequential(
            Conv2d(in_channels=64,
                   out_channels=num_classes,
                   kernel_size=(1, 1),
                   stride=(1, 1)),
            BatchNorm2d(num_features=num_classes),
            ReLU()
        )
        
        self.batch_norm = BatchNorm2d(num_classes)
        self.global_pooling = AdaptiveAvgPool2d(1)  # Global average pooling
        #self.linear = Linear(in_features=10, out_features=10)
    

    def forward(self, 
                X: Tensor) -> Tensor:
        X = X.float()
        X = X if X.ndimension() == 4 else X.unsqueeze(1)
        
        # ResNet
        out_res_1 = self.res_net_1(X[:, :, 0:64, :])
        out_res_2 = self.res_net_2(X[:, :, 64:128, :])
        out = cat((out_res_1, out_res_2), axis=2)

        out = self.conv_out_1(out)
        out = self.conv_out_2(out)
        
        out = self.batch_norm(out)
        
        out = self.global_pooling(out)
        out = out.squeeze()

        return out

if __name__ == "__main__":
    dnn = ResNet_d()
    x = rand(4, 3, 128, 423) 
    y = dnn(x)
    print(y.shape)   