from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet264', 'densenet29', 'densenet45',
           'densenet85']


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features,
                                           bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([input, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    growth_rate (int) - how many filters to add each layer (`k` in paper)
    block_config (list of 4 ints) - how many layers in each pooling block
    num_init_features (int) - the number of filters to learn in the first convolution layer
    bn_size (int) - multiplicative factor for number of bottle neck layers
      (i.e. bn_size * k features in the bottleneck layer)
    drop_rate (float) - dropout rate after each dense layer
    num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 num_init_feature=24, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # Firsrt convolution before dense block
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_feature, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_feature)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))])
        )

        num_features = num_init_feature
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Conv2d(num_features, num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        features = self.features(input)
        out = self.classifier(features).view(input.size(0), -1)
        return out


def densenet121(pretrained=False, **kwargs):
    model = DenseNet(num_init_feature=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(pretrained=False, **kwargs):
    model = DenseNet(num_init_feature=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(pretrained=False, **kwargs):
    model = DenseNet(num_init_feature=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet264(pretrained=False, **kwargs):
    model = DenseNet(num_init_feature=64, growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)
    return model


class DenseNet_CIFAR10(nn.Module):
    """
    growth_rate (int) - how many filters to add each layer (`k` in paper)
    block_config (list of 4 ints) - how many layers in each pooling block
    num_init_features (int) - the number of filters to learn in the first convolution layer
    bn_size (int) - multiplicative factor for number of bottle neck layers
      (i.e. bn_size * k features in the bottleneck layer)
    drop_rate (float) - dropout rate after each dense layer
    num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 12),
                 num_init_feature=24, bn_size=4, drop_rate=0, num_classes=10):
        super(DenseNet_CIFAR10, self).__init__()

        # Firsrt convolution before dense block
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_feature, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_feature)),
            ('relu0', nn.ReLU(inplace=True))])
        )

        num_features = num_init_feature
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Conv2d(num_features, num_classes,kernel_size=1,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                ps = list(m.parameters())
                if len(ps) == 2:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)



    def forward(self, input):
        features = self.features(input)
        out = self.classifier(features).view(input.size(0), -1)
        return out


def densenet29(pretrained=False, **kwargs):
    model = DenseNet_CIFAR10(num_init_feature=48, growth_rate=24, block_config=(6, 6, 6, 6), **kwargs)
    return model


def densenet45(pretrained=False, **kwargs):
    model = DenseNet_CIFAR10(num_init_feature=48, growth_rate=24, block_config=(10, 10, 10, 10), **kwargs)
    return model


def densenet85(pretrained=False, **kwargs):
    model = DenseNet_CIFAR10(num_init_feature=48, growth_rate=24, block_config=(20, 20, 20, 20), **kwargs)
    return model


if __name__ == '__main__':
    net = densenet29().to("cuda:0")
    import torchsummary

    torchsummary.summary(net, input_size=(3, 32, 32))
