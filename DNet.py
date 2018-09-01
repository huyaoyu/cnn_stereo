import torch.nn as nn
import math


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier_d = nn.Sequential(
            # nn.Linear(512 * 5 * 10, 1024),
            # nn.ReLU(True),
            # # nn.Dropout(),
            # nn.Linear(1024, 256),
            # nn.ReLU(True),
            # # nn.Dropout(),
            # nn.Linear(256, 1),
            # nn.Sigmoid()
            nn.Linear(512,128),
            nn.LeakyReLU(True),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.avg_pool2d(x,kernel_size=(5,10))
        x = x.view(x.size(0), -1)
        x = self.classifier_d(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 4
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 'M', 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def DNet(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=False), **kwargs)
