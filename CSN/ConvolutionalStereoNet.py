
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_NET_DICT = {
    "convNets_FX": [
        {"inChannels":   3, "outChannels":  32, "kernelSize": 3, "stride": 1, "padding": 1},
        {"inChannels":  32, "outChannels":  32, "kernelSize": 3, "stride": 1, "padding": 1},
        {"inChannels":  32, "outChannels":  64, "kernelSize": 3, "stride": 1, "padding": 1},
        {"inChannels":  64, "outChannels":  64, "kernelSize": 3, "stride": 1, "padding": 1},
        {"inChannels":  64, "outChannels": 128, "kernelSize": 3, "stride": 1, "padding": 1},
        {"inChannels": 128, "outChannels": 128, "kernelSize": 3, "stride": 1, "padding": 1}
    ],
    "convNets_Exp": [
        {"inChannels": 128, "outChannels": 128, "kernelSize": 3, "stride": 2, "padding": 1},
        {"inChannels": 128, "outChannels": 256, "kernelSize": 3, "stride": 2, "padding": 1},
        {"inChannels": 256, "outChannels": 256, "kernelSize": 3, "stride": 2, "padding": 1},
        {"inChannels": 256, "outChannels": 512, "kernelSize": 3, "stride": 2, "padding": 1}
    ],
    "convNets_Cnt": [
        {"inChannels": 512, "outChannels": 256, "kernelSize": 3, "stride": 2, "padding": 1},
        {"inChannels": 256, "outChannels": 256, "kernelSize": 3, "stride": 2, "padding": 1},
        {"inChannels": 256, "outChannels": 128, "kernelSize": 3, "stride": 2, "padding": 1},
        {"inChannels": 128, "outChannels": 128, "kernelSize": 3, "stride": 2, "padding": 1},
    ],
    "convNets_DR": [
        {"inChannels": 128, "outChannels":  64, "kernelSize": 1, "stride": 1, "padding": 0},
        {"inChannels":  64, "outChannels":   1, "kernelSize": 1, "stride": 1, "padding": 0},
    ]
}

class ConvolutionalStereoNet(nn.Module):
    def __init__(self):
        super(ConvolutionalStereoNet, self).__init__()

        # Declare each layer.
        convNets_FX = CONV_NET_DICT["convNets_FX"]
        self.fx     = []
        self.fx_bn  = []

        # Feature extraction layers.
        for fxd in convNets_FX:
            self.fx.append( \
                nn.Conv2d( fxd["inChannels"],\
                    fxd["outChannels"],\
                    fxd["kernelSize"],\
                    stride = fxd["stride"],\
                    padding = fxd["padding"] ) )

            self.fx_bn.append( nn.BatchNorm2d( fxd["outChannels"] ) )
        
        # Expansion layers.
        convNets_Exp = CONV_NET_DICT["convNets_Exp"]
        self.exp     = []

        for expd in convNets_Exp:
            self.exp.append( \
                nn.Conv2d( expd["inChannels"],\
                    expd["outChannels"],\
                    expd["kernelSize"],\
                    stride = expd["stride"],\
                    padding = expd["padding"] ) )

        # Contraction layers.
        convNets_Cnt = CONV_NET_DICT["convNets_Cnt"]
        self.cnt     = []

        for cntd in convNets_Cnt:
            self.cnt.append( \
                nn.ConvTranspose2d( cntd["inChannels"],\
                    cntd["outChannels"],\
                    cntd["kernelSize"],\
                    stride = cntd["stride"],\
                    padding = cntd["padding"] ) )
    
        # Depth reconstruction layers.
        convNets_DR = CONV_NET_DICT["convNets_DR"]
        self.dr = []

        for drd in convNets_DR:
            self.dr.append( \
                nn.Conv2d( drd["inChannels"],\
                    drd["outChannels"],\
                    drd["kernelSize"],\
                    drd["stride"],\
                    drd["padding"] ) )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.fx:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, 0, math.sqrt(2. / n))

            if m.bias is not None:
                m.bias.data.zero_()
        
        for m in self.exp:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, 0, math.sqrt(2. / n))

            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.cnt:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_( m.weight, 0, math.sqrt( 2.0 / n) )
            
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.dr:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_( m.weight, 0, math.sqrt( 2.0 / n) )
            
            if m.bias is not None:
                m.bias.data.zero_()
        
        for m in self.fx_bn:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def feature_extract(self, x):
        # Manually run through each layer!!!
        out  = self.fx[0](x)
        out  = self.fx_bn[0](out)
        out  = F.relu(out, inplace = True)

        out  = self.fx[1](x)
        cat1 = self.fx_bn[1](out)
        out  = F.max_pool2d( F.relu( cat1, inplace = False ), kernel_size = 2 )

if __name__ == "__main__":
    # Create a ConvolutionalStereoNet object.

    csn = ConvolutionalStereoNet()
    