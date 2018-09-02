
from __future__ import print_function

import cv2
import math
import numpy as np

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
    "convNets_FX_Cat": [
        {"inChannels":  64, "outChannels":  32, "kernelSize": 1, "stride": 1, "padding": 0},
        {"inChannels": 128, "outChannels":  32, "kernelSize": 1, "stride": 1, "padding": 0}
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

        convNets_FX_Cat = CONV_NET_DICT["convNets_FX_Cat"]
        self.fx_cat = []

        for fx_cat_d in convNets_FX_Cat:
            self.fx_cat.append( \
                nn.Conv2d( fx_cat_d["inChannels"],\
                    fx_cat_d["outChannels"],\
                    fx_cat_d["kernelSize"],\
                    fx_cat_d["stride"],
                    fx_cat_d["padding"]) )

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
        
        for m in self.fx_cat:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, 0, math.sqrt(2. / n))

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

        out  = self.fx[1](out)
        cat1 = self.fx_bn[1](out)
        out  = F.max_pool2d( F.relu( cat1, inplace = False ), kernel_size = 2 )

        out  = self.fx[2](out)
        out  = self.fx_bn[2](out)
        out  = F.relu(out, inplace = True)

        out  = self.fx[3](out)
        cat2 = self.fx_bn[3](out)
        out  = F.max_pool2d( F.relu( cat2, inplace = False ), kernel_size = 2 )
        cat2 = F.interpolate( cat2, scale_factor = 2, mode = "bilinear", align_corners = True )
        cat2 = self.fx_cat[0](cat2)

        out  = self.fx[4](out)
        out  = self.fx_bn[4](out)
        out  = F.relu(out, inplace = True)

        out  = self.fx[5](out)
        cat3 = self.fx_bn[5](out)
        cat3 = F.interpolate( cat3, scale_factor = 4, mode = "bilinear", align_corners = True )
        cat3 = self.fx_cat[1](cat3)

        x = torch.cat((x, cat1, cat2, cat3), dim = 1)

        return x


if __name__ == "__main__":
    # Create a ConvolutionalStereoNet object.

    csn = ConvolutionalStereoNet()

    # Load a test image.
    img = cv2.imread("../data/airsim_oldtown_stereo_01/image/000000_210520_0_rgb.png")

    nImg = img.astype(np.float32) / np.max(img)

    t = torch.from_numpy(nImg.transpose(2, 0, 1)).view(1, img.shape[2], img.shape[0], img.shape[1])

    import ipdb; ipdb.set_trace()

    # Test feature extraction.
    x = csn.feature_extract(t)
    