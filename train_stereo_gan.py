import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from torch.logger import Logger
from StereoNet3 import StereoNet3 as StereoNet
from DNet25 import DNet25 as DNet
from sceneflowDataset import SceneflowDataset
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(precision=4, threshold=np.nan)

exp_prefix = '12-3-4_'
stereo_lr = 1e-5
dnet_lr = 1e-5
batch = 1
trainstep = 100000
showiter = 10
snapshot = 10000
paramName = 'models/'+exp_prefix+'stereo_gan'
predModel = 'models/12-3-3_stereo_gan_100000.pkl'
dnetPreModel = 'models/12-3-3_stereo_gan_dnet_100000.pkl'
lamb = 1

stereonet = StereoNet()
stereonet.cuda()
loadPretrain(stereonet,predModel)

print '---'
dnet = DNet()
dnet.cuda()
loadPretrain(dnet,dnetPreModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
sceneDataset = SceneflowDataset(transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((7,37,37)),
													ToTensor(),
													normalize]))

dataloader = DataLoader(sceneDataset, batch_size=batch, shuffle=True, num_workers=8)


criterion1 = nn.SmoothL1Loss()
# stereoOptimizer = optim.Adam(stereonet.parameters(), lr = Lr)
stereoOptimizer = optim.Adam([{'params':stereonet.preLoadedParams,'lr': stereo_lr},
								{'params':stereonet.params}], lr = stereo_lr)

criterion2 = nn.BCELoss()
dnetOptimizer = optim.Adam(dnet.parameters(), lr = dnet_lr)

label = torch.FloatTensor(batch)
real_label = 1
fake_label = 0
label = Variable(label)
label = label.cuda()

logger = Logger('./logs-gan')

lossplot = []
d_loss_plot = []
fake_d_loss_plot = []
real_d_loss_plot = []
g_loss_plot = []
running_loss = 0.0
running_d_loss = 0.0
running_fake_d_loss = 0.0
running_real_d_loss = 0.0
running_fake_d_mean = 0.0
running_real_d_mean = 0.0
running_fake_d_mean2 = 0.0
running_g_loss = 0.0

ind = 0
while True:

	for sample in dataloader:

		ind = ind+1

		leftTensor = sample['left']
		rightTensor = sample['right']
		targetdisp = sample['disp']

		dnetOptimizer.zero_grad()
		dnet.zero_grad()
		# train with real
		inputVariable = Variable(targetdisp,requires_grad=True)
		output = dnet(inputVariable.cuda())
		label.data.resize_(batch).fill_(real_label)
		errD_real = criterion2(output, label)
		errD_real.backward()
		D_x = output.data.mean()

		# train with fake
		fake = stereonet(Variable(leftTensor.cuda(),requires_grad=True),Variable(rightTensor.cuda(),requires_grad=True))
		label.data.fill_(fake_label)
		# inputVariable = Variable(fake.detach().data,requires_grad=True)
		# print inputVariable.size()
		output = dnet(fake.detach())
		errD_fake = criterion2(output, label)
		errD_fake.backward()
		D_G_z1 = output.data.mean()
		errD = errD_real + errD_fake
		# if errD.data[0]>0.7:
		dnetOptimizer.step()

		stereoOptimizer.zero_grad()
		stereonet.zero_grad()

		errSup = criterion1(fake,Variable(targetdisp.cuda()))
		# if ind%5==1:
		errSup.backward(retain_graph=True)
		# else:
		# 	errSup.backward()

		# if ind%5==1:
		label.data.fill_(real_label)  # fake labels are real for generator cost
		output = dnet(fake)
		errG = lamb*criterion2(output, label)
		errG.backward()
		D_G_z2 = output.data.mean()

		stereoOptimizer.step()

		running_loss += errSup.data[0]
		running_d_loss += errD.data[0]
		running_fake_d_loss += errD_fake.data[0]
		running_real_d_loss += errD_real.data[0]
		running_fake_d_mean += D_G_z1
		running_real_d_mean += D_x
		running_fake_d_mean2 += D_G_z2
		running_g_loss += errG.data[0]
		if ind % showiter == 0:    # print every 20 mini-batches
			timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
			print('%s[%d %s] stereo_loss:%.5f, g_loss:%.5f, d_loss:%.5f, D(x):%.5f (%.5f) D(G):%.5f (%.5f /%.5f)' %
			(exp_prefix[0:-1], ind, timestr, running_loss / showiter, running_g_loss/showiter, running_d_loss/showiter,
				running_real_d_loss/showiter, running_real_d_mean/showiter,
				running_fake_d_loss/showiter, running_fake_d_mean/showiter, running_fake_d_mean2/showiter))
			logger.scalar_summary('loss-gan-stereo',running_loss/showiter,ind)
			logger.scalar_summary('loss-gan-d',running_d_loss/showiter,ind)
			logger.scalar_summary('loss-gan-d-real',running_real_d_loss/showiter,ind)
			logger.scalar_summary('loss-gan-d-fake',running_fake_d_loss/showiter,ind)
			logger.scalar_summary('loss-gan-g',running_g_loss/showiter,ind)
			running_loss = 0.0
			running_d_loss = 0.0
			running_fake_d_loss = 0.0
			running_real_d_loss = 0.0
			running_fake_d_mean = 0.0
			running_real_d_mean = 0.0
			running_fake_d_mean2 = 0.0
			running_g_loss = 0.0
		lossplot.append(errSup.data[0])
		d_loss_plot.append(errD.data[0])
		fake_d_loss_plot.append(errD_fake.data[0])
		real_d_loss_plot.append(errD_real.data[0])
		g_loss_plot.append(errG.data[0])
		# print loss.data[0]

		if (ind)%snapshot==0:
			torch.save(stereonet.state_dict(), paramName+'_'+str(ind)+'.pkl')
			torch.save(dnet.state_dict(),paramName+'_dnet_'+str(ind)+'.pkl')

		if ind==trainstep:
			break
	if ind==trainstep:
		break

lossfilename = exp_prefix+'loss'
import matplotlib.pyplot as plt
group = 100
lossplot = np.array(lossplot)
from scipy.io import savemat
savemat(lossfilename+'.mat',{'loss':lossplot})
if len(lossplot)%group>0:
	lossplot = lossplot[0:len(lossplot)/group*group]
lossplot = lossplot.reshape((-1,group))
lossplot = lossplot.mean(axis=1)
plt.plot(lossplot)
# plt.ylim([0,0.5])
plt.savefig(lossfilename,pad_inches=0)
plt.show()

