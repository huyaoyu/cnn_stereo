import torch.nn as nn
import math

# 640x320 -- 
# 320x160 --
# 160x80  --
# 80x40   
# 40x20   --
# 20x10
# 10x5    --
# (in-k+2p)/s+1
class DNet25(nn.Module):
    """ Enhance the dnet with two extra conv layers and more channels
    """
    def __init__(self):
        super(DNet25, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32, 32, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32, 64, kernel_size=4, stride = 4, padding=0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64, 128, kernel_size=4, stride = 4, padding=0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128, 128, kernel_size=5, stride = 5, padding=0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128, 64, kernel_size=(1, 2), stride = 1, padding=0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64, 1, kernel_size=1, stride = 1, padding=0),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
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

class DNet252(nn.Module):
    """ Enhance the dnet with two extra conv layers and more channels
    """
    def __init__(self):
        super(DNet252, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32, 32, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32, 64, kernel_size=4, stride = 4, padding=0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64, 128, kernel_size=4, stride = 4, padding=0),
            nn.LeakyReLU(0.2,True),
            # nn.Conv2d(128, 128, kernel_size=5, stride = 5, padding=0),
            # nn.LeakyReLU(0.2,True),
            # nn.Conv2d(128, 64, kernel_size=(1, 2), stride = 1, padding=0),
            # nn.LeakyReLU(0.2,True),
            # nn.Conv2d(64, 1, kernel_size=1, stride = 1, padding=0),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        return x


# import torch
# from torch.autograd import Variable
# dnet = DNet25()
# dnet.cuda()
# import numpy as np
# import matplotlib.pyplot as plt
# np.set_printoptions(precision=4, threshold=np.nan)
# imsize = 640
# img = np.random.normal(size=(320,640,1))
# img = img.astype(np.float32)
# print img.dtype

# imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
# imgTensor = torch.from_numpy(imgInput)
# print imgTensor.size()
# z = dnet(Variable(imgTensor.cuda(),requires_grad=False))
# print z.data.cpu().numpy().shape
# print z.data.cpu().numpy()
