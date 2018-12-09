######################################################################################################
######################################################################################################
######################################################################################################
####################################   Identity mapping ResNet   #####################################
######################################################################################################
######################################################################################################
######################################################################################################
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__=['ResNet','resnet18','resnet34','resnet50','resnet101','resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes,out_planes,stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

def conv1x1(in_planes,out_planes,stride=1):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()

        self.bn1=nn.BatchNorm2d(inplanes)
        self.conv1=conv3x3(inplanes,planes,stride)
        
        self.relu=nn.ReLU(inplace=True)

        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=conv3x3(planes,planes)
        
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)

        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)
        
        if self.downsample is not None:
            residual=self.downsample(x)

        out+=residual

        return out

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.bn1=nn.BatchNorm2d(inplanes)
        self.conv1=conv1x1(inplanes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=conv3x3(planes,planes,stride)
        self.bn3 = nn.BatchNorm2d(planes )
        self.conv3=conv1x1(planes,planes*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x
        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)
        

        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)

        out=self.bn3(out)
        out=self.relu(out)
        out=self.conv3(out)
        
        if self.downsample is not None:
            residual=self.downsample(x)

        out+=residual
        

        return out

class ResNet_CIFAR10(nn.Module):
    
    def __init__(self, block,layers,num_classes=10):
        super(ResNet_CIFAR10, self).__init__()
        self.inplanes=16
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1=self._make_layer(block,16,layers[0])
        self.layer2=self._make_layer(block,32,layers[1],stride=2)
        self.layer3=self._make_layer(block,64,layers[2],stride=2)
        self.bn3=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(64*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride),
                nn.BatchNorm2d(planes*block.expansion),
                )

        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        x=self.bn3(x)
        x=self.relu(x)

        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x

def resnet20(pretrained=False,num_class=10):
    model=ResNet_CIFAR10(BasicBlock,[3,3,3,3],num_class)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet20']))
    return model


def resnet32(pretrained=False,num_class=10):
    model=ResNet_CIFAR10(BasicBlock,[5,5,5,5],num_class)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet32']))
    return model


def resnet44(pretrained=False,num_class=10):
    model=ResNet_CIFAR10(Bottleneck,[7,7,7,7],num_class)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet44']))
    return model

def resnet110(pretrained=False,num_class=10):
    model=ResNet_CIFAR10(Bottleneck,[18,18,18,18],num_class)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet110']))
    return model

def resnet302(pretrained=False,num_class=10):
    model=ResNet_CIFAR10(Bottleneck,[50,50,50,50],num_class)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet302']))
    return model
