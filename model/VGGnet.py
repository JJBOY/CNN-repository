import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__help__="you can call VGGnet(kind='vgg16',num_classes=1000,batch_norm=False,pretrained=False) to get a vgg net,\
         you can use __all__ to get the compelete vggnet choose.\
         if you want to use vggxx_bn you should not give the parameter kind='vggxx_bn',\
         you should also give the kind='vggxx_bn' but another parameter batch_norm=True"

__all__=[
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    
    def __init__(self,features,num_classes=1000,init_weights=True):
        super(VGG, self).__init__()
        self.features=features
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
            )
        self.conv1x1=nn.Conv2d(512,num_classes,kernel_size=1,stride=1)
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x=self.features(x)
        x=self.conv1x1(x)
        x=x.view(x.size(0),-1)
        #x=self.classifier(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

cfg={
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'], #11 weight layers
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'], #13 weight layers
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'], #16 weight layers
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'], #19 weight layers
}
def make_layers(cfg,batch_norm=False):
    layers=[]
    in_channels=3
    for v in cfg:
        if v=='M':
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            if batch_norm:
                layers+=[nn.Conv2d(in_channels,v,kernel_size=3,padding=1,stride=1,bias=False),
                        nn.BatchNorm2d(v),nn.ReLU(True)]
            else:
                layers+=[nn.Conv2d(in_channels,v,kernel_size=3,padding=1,stride=1),nn.ReLU(True)]
            in_channels=v
    return nn.Sequential(*layers)

def VGGnet(kind='vgg16',num_classes=1000,batch_norm=False,pretrained=False,**kwargs):
    if pretrained:
        kwargs['init_weights']=False
        assert num_classes==1000,\
            'pretrained model only on ImageNet which num classes is 1000 but got{}'.format(num_classes)
    model=VGG(make_layers(cfg[kind],batch_norm),num_classes,**kwargs)
    if pretrained:
        name=kind
        if batch_norm==True:
            name+='_bn'
        model.load_state_dict(model_zoo.load_url(model_urls[name]))
    return model

if __name__ == '__main__':
    a=nn.Conv2d(1,2,kernel_size=1,bias=False)
    print(a.bias)
    #model=VGGnet(kind='vgg16',num_classes=10,batch_norm=True)
    #print(model)

