import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from torch.logger import Logger
from StereoNet3 import StereoNet3 as StereoNet
from sceneflowDataset import SceneflowDataset
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import savemat
np.set_printoptions(precision=4, threshold=np.nan)

exp_prefix = '12-0_'
Lr = 1e-4
batch = 1
trainstep = 100000
showiter = 20
snapshot = 10000
paramName = 'models/'+exp_prefix+'stereo_2'
predModel = 'models/9-2-2_stereo_2_50000.pkl'
lossfilename = exp_prefix+'loss'

stereonet = StereoNet()
stereonet.cuda()
# loadPretrain(stereonet,predModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
sceneDataset = SceneflowDataset(transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((7,37,37)),
													ToTensor(),
													normalize]))
dataloader = DataLoader(sceneDataset, batch_size=batch, shuffle=True, num_workers=8)


criterion = nn.SmoothL1Loss()
# stereoOptimizer = optim.Adam(stereonet.parameters(), lr = Lr)
stereoOptimizer = optim.Adam([{'params':stereonet.preLoadedParams,'lr':Lr},
								{'params':stereonet.params}], lr = Lr)

lossplot = []
running_loss = 0.0

ind = 0

logger = Logger('./logs')

while True:

	for sample in dataloader:

		ind = ind+1

		leftTensor = sample['left']
		rightTensor = sample['right']
		targetdisp = sample['disp']

		stereoOptimizer.zero_grad()

		# forward + backward + optimize
		output = stereonet(Variable(leftTensor.cuda(),requires_grad=True),Variable(rightTensor.cuda(),requires_grad=True))
		loss = criterion(output, Variable(targetdisp.cuda()))

		running_loss += loss.data[0]
		if ind % showiter == 0:    # print every 20 mini-batches
			timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
			print(exp_prefix[0:-1] + ' [%d %s] loss: %.5f lr: %f' %
			(ind, timestr, running_loss / showiter, stereoOptimizer.param_groups[1]['lr']))
			# add to tensorboard
			logger.scalar_summary('loss-2',running_loss/showiter,ind)

			running_loss = 0.0
		lossplot.append(loss.data[0])
		# print loss.data[0]

		loss.backward()
		stereoOptimizer.step()

		if (ind)%snapshot==0:
			torch.save(stereonet.state_dict(), paramName+'_'+str(ind)+'.pkl')
			savemat(lossfilename+'.mat',{'loss':np.array(lossplot)})

		if ind==trainstep:
			break
	if ind==trainstep:
		break

import matplotlib.pyplot as plt
group = 100
lossplot = np.array(lossplot)
if len(lossplot)%group>0:
	lossplot = lossplot[0:len(lossplot)/group*group]
lossplot = lossplot.reshape((-1,group))
lossplot = lossplot.mean(axis=1)
plt.plot(lossplot)
# plt.ylim([0,0.5])
plt.savefig(lossfilename,pad_inches=0)
plt.show()

