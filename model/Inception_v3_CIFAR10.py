# Reference:
# "Rethinking the Inception Architecture for Computer Vision"

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.model_zoo as model_zoo


class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x

class InceptionA(nn.Module):
    '''
    对应文中的figure 5.b把5x5的卷积分解成两个3x3的卷积.
    factories the 5x5 convlolution operation into 2 3x3 convolutions.
    refer to the figure 5 in the paper.
    '''
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1=BasicConv2d(in_channels,32,kernel_size=1)

        self.branch3x3_1=BasicConv2d(in_channels,24,kernel_size=1)
        self.branch3x3_2=BasicConv2d(24,32,kernel_size=3,padding=1)

        self.branch5x5_1=BasicConv2d(in_channels,32,kernel_size=1)
        self.branch5x5_2=BasicConv2d(32,48,kernel_size=3,padding=1)
        self.branch5x5_3=BasicConv2d(48,48,kernel_size=3,padding=1)

        self.branch_pool_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.branch_pool_2=BasicConv2d(in_channels,32,kernel_size=1)

    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)
        branch5x5=self.branch5x5_3(branch5x5)

        branch_pool=self.branch_pool_1(x)
        branch_pool=self.branch_pool_2(branch_pool)

        outputs=[branch5x5,branch3x3,branch_pool,branch1x1]

        return torch.cat(outputs,1)



class InceptionB(nn.Module):
    '''
    对应文中的figure 6.把nxn分解成1xn和nx1.文中n=17
    factories the nxn convolution to 1xn and nx1. refer to the figure 6 in the paper
    '''
    def __init__(self,in_channels,channels_7x7):
        super(InceptionB, self).__init__()
        self.branch1x1=BasicConv2d(in_channels,192//2,kernel_size=1)

        c7=channels_7x7
        self.branch7x7_1=BasicConv2d(in_channels,c7,kernel_size=1)
        self.branch7x7_2=BasicConv2d(c7,c7,kernel_size=(1,7),padding=(0,3))
        self.branch7x7_2=BasicConv2d(c7,192//2,kernel_size=(7,1),padding=(3,0))

        self.branch13x13_1=BasicConv2d(in_channels,c7,kernel_size=1)
        self.branch13x13_2=BasicConv2d(c7,c7,kernel_size=(1,7),padding=(0,3))
        self.branch13x13_3=BasicConv2d(c7,c7,kernel_size=(7,1),padding=(3,0))
        self.branch13x13_4=BasicConv2d(c7,c7,kernel_size=(1,7),padding=(0,3))
        self.branch13x13_5=BasicConv2d(c7,192//2,kernel_size=(7,1),padding=(3,0))

        self.branch_pool_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.branch_pool_2=BasicConv2d(in_channels,192//2,kernel_size=1)

    def forward(self,x):
        branch13x13=self.branch13x13_1(x)
        branch13x13=self.branch13x13_2(branch13x13)
        branch13x13=self.branch13x13_3(branch13x13)
        branch13x13=self.branch13x13_4(branch13x13)
        branch13x13=self.branch13x13_5(branch13x13)

        branch7x7=self.branch7x7_1(x)
        branch7x7=self.branch7x7_2(branch7x7)

        branch_pool=self.branch_pool_1(x)
        branch_pool=self.branch_pool_2(branch_pool)

        branch1x1=self.branch1x1(x)

        outputs=[branch13x13,branch7x7,branch_pool,branch1x1]

        return torch.cat(outputs,1)

class InceptionC(nn.Module):
    '''
    对应文中的figure 7.
    refer to the figure 7 in the paper
    '''
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        
        self.branch3x5_1=BasicConv2d(in_channels,448//2,kernel_size=1)
        self.branch3x5_2=BasicConv2d(448//2,384//2,kernel_size=3,padding=1)
        self.branch3x5_3=BasicConv2d(384//2,384//2,kernel_size=(1,3),padding=(0,1))
        self.branch3x5_4=BasicConv2d(384//2,384//2,kernel_size=(3,1),padding=(1,0))

        self.branch3x3_1=BasicConv2d(in_channels,384//2,kernel_size=1)
        self.branch3x3_2=BasicConv2d(384//2,384//2,kernel_size=(1,3),padding=(0,1))
        self.branch3x3_3=BasicConv2d(384//2,384//2,kernel_size=(3,1),padding=(1,0))

        self.branch_pool_1=nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.branch_pool_2=BasicConv2d(in_channels,192//2,kernel_size=1)

        self.branch1x1=BasicConv2d(in_channels,320//2,kernel_size=1)

    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch3x3=self.branch3x3_1(x)
        branch3x3=torch.cat([self.branch3x3_2(branch3x3),self.branch3x3_3(branch3x3)],1)

        branch3x5=self.branch3x5_1(x)
        branch3x5=self.branch3x5_2(branch3x5)
        branch3x5=torch.cat([self.branch3x5_3(branch3x5),self.branch3x5_4(branch3x5)],1)

        branch_pool=self.branch_pool_1(x)
        branch_pool=self.branch_pool_2(branch_pool)

        outputs=[branch3x5,branch3x3,branch_pool,branch1x1]
        return torch.cat(outputs,1)

class InceptionD(nn.Module):
    '''
    对应文中的figure 10.改进的pooling操作.
    improved pooling operation. refer to the figure 10 in the paper
    '''
    def __init__(self,in_channels,out_channels=64,add_channels=0):
        super(InceptionD, self).__init__()

        self.branch3x3_1=BasicConv2d(in_channels,out_channels//2,kernel_size=1)
        self.branch3x3_2=BasicConv2d(out_channels//2,302//2+add_channels//2,kernel_size=3,stride=2)
        
        self.branch5x5_1=BasicConv2d(in_channels,out_channels//2,kernel_size=1)
        self.branch5x5_2=BasicConv2d(out_channels//2,178//2+add_channels//2,kernel_size=3,padding=1)
        self.branch5x5_3=BasicConv2d(178//2+add_channels//2,178//2+add_channels//2,kernel_size=3,stride=2)

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2)

    def forward(self,x):

        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)
        branch5x5=self.branch5x5_3(branch5x5)

        branch_pool=self.maxpool(x)

        outputs=[branch5x5,branch3x3,branch_pool]

        return torch.cat(outputs,1)



class Inceptionv3(nn.Module):
    def __init__(self, num_classes=10,aux_logits=True):
        super(Inceptionv3, self).__init__()
        self.aux_logits=aux_logits

        
        self.Conv3=BasicConv2d(3,64,kernel_size=3,padding=1)
        self.Conv4=BasicConv2d(64,144,kernel_size=3,padding=1)

        self.Mixed_5a=InceptionA(144)
        self.Mixed_5b=InceptionA(144)
        self.Mixed_pool1=InceptionD(in_channels =144)
        self.Mixed_6a=InceptionB(384,channels_7x7=64)
        self.Mixed_6b=InceptionB(384,channels_7x7=80)
        self.Mixed_6c=InceptionB(384,channels_7x7=80)
       

        if aux_logits:
            self.AuxLogits=InceptionAux(384,num_classes)

        self.Mixed_pool2=InceptionD(384,81,16)
        self.Mixed_7a=InceptionC(640)
        self.Mixed_7b=InceptionC(1024)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Conv2d(1024,num_classes,kernel_size=1,stride=1)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,x):

        aux=None

        x=self.Conv3(x)
        x=self.Conv4(x)

        x=self.Mixed_5a(x)
        x=self.Mixed_5b(x)
        
        x=self.Mixed_pool1(x)

        x=self.Mixed_6a(x)
        x=self.Mixed_6b(x)
        x=self.Mixed_6c(x)

        if self.aux_logits:
            aux = self.AuxLogits(x)

        x=self.Mixed_pool2(x)

        x=self.Mixed_7a(x)
        x=self.Mixed_7b(x)

        x=self.avgpool(x)

        x=self.fc(x)
        x=x.view(x.size(0),-1)

        return x,aux


class InceptionAux(nn.Module):
    """docstring for InceptionAux"""
    def __init__(self, in_channels,num_classes=10):
        super(InceptionAux, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Conv2d(in_channels,num_classes,kernel_size=1,stride=1)

    def forward(self,x):
        x=self.avgpool(x)
        x=self.fc(x)
        x=x.view(x.size(0),-1)
        return x


        
if __name__ == '__main__':
    model=Inceptionv3()
    input=torch.Tensor(1,3,32,32)
    print(model)
    print(model(input))
        

