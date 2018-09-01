import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from torch.logger import Logger
from StereoNet3 import StereoNet3 as StereoNet
from sceneflowDataset import SceneflowDataset,sample2img
from kittiDataset import KittiDataset
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import savemat
np.set_printoptions(precision=4, threshold=np.nan)

exp_prefix = '9-5_'
Lr = 1e-5
batch = 2
trainstep = 50000
showiter = 20
snapshot = 10000
paramName = 'models/'+exp_prefix+'stereo_2'
predModel = 'models/9-3_stereo_2_100000.pkl'
lossfilename = exp_prefix+'loss'
SceneTurn = 5
ImgHeight = 320
ImgWidth = 640

stereonet = StereoNet()
stereonet.cuda()
loadPretrain(stereonet,predModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

sceneDataset = SceneflowDataset(transform=Compose([ RandomCrop(size=(ImgHeight,ImgWidth)),
													RandomHSV((10,80,80)),
													ToTensor(),
													normalize]))
kittiDataset = KittiDataset(transform=Compose([ RandomCrop(size=(ImgHeight,ImgWidth)),
													RandomHSV((7,50,50)),
													ToTensor(),
													normalize]),
							surfix='train')

sceneDataloader = DataLoader(sceneDataset, batch_size=batch, shuffle=True, num_workers=4)
kittiDataloader = DataLoader(kittiDataset, batch_size=batch, shuffle=True, num_workers=4)
sceneiter = iter(sceneDataloader)
kittiiter = iter(kittiDataloader)

criterion = nn.SmoothL1Loss()
# stereoOptimizer = optim.Adam(stereonet.parameters(), lr = Lr)
stereoOptimizer = optim.Adam([{'params':stereonet.preLoadedParams,'lr':Lr},
								{'params':stereonet.params}], lr = Lr)

lossplot = []
running_loss = 0.0

ind = 0
sceneCount = 0
logger = Logger('./logs')
while True:

	ind = ind+1

	if sceneCount<SceneTurn:
		# print 'load scene..'
		sceneCount = sceneCount + 1
		try:
			sample = sceneiter.next()
		except StopIteration:
			sceneiter = iter(sceneDataloader)
			sample = sceneiter.next()
	else:
		# print 'load kitti'
		sceneCount = 0
		try:
			sample = kittiiter.next()
		except StopIteration:
			kittiiter = iter(kittiDataloader)
			sample = kittiiter.next()

	leftTensor = sample['left']
	rightTensor = sample['right']
	targetdisp = sample['disp']

	mask = (targetdisp<=0).cuda().float() # in kitti dataset, value -1 is set to unmesured pixels
	mask2 = (targetdisp>0).cuda().float()
	validPixel = mask2.sum()
	lossrate = ImgWidth*ImgHeight*batch/validPixel
	# print '  lossrate:', lossrate

	stereoOptimizer.zero_grad()

	# forward + backward + optimize
	output = stereonet(Variable(leftTensor.cuda(),requires_grad=True),Variable(rightTensor.cuda(),requires_grad=True))
	output2 = output * Variable(mask2, requires_grad=False) + Variable(targetdisp.cuda(), requires_grad=False) * Variable(mask, requires_grad=False) # filter for kitti spart GT
	loss = criterion(output2, Variable(targetdisp.cuda()))*lossrate

	running_loss += loss.data[0]
	if ind % showiter == 0:    # print every 20 mini-batches
		timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
		print(exp_prefix[0:-1] + ' [%d %s] loss: %.5f lr: %f' %
		(ind, timestr, running_loss / showiter, stereoOptimizer.param_groups[1]['lr']))
		# add to tensorboard
		logger.scalar_summary('loss-kitti',running_loss/showiter,ind)

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

