import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
import torch.optim as optim

from StereoNet3 import StereoNet3 as StereoNet
from DNet3 import DNet3 as DNet
from sceneflowDataset import SceneflowDataset
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(precision=4, threshold=np.nan)

exp_prefix = '13-1_'
stereo_lr = 0
dnet_lr = 0
batch = 1
trainstep = 100000
showiter = 1
snapshot = 10000
paramName = 'models/'+exp_prefix+'stereo_gan'
predModel = 'models/9-3_stereo_2_100000.pkl'
# dnetPreModel = 'models/4-2_dnet_1000.pkl'
LAMBDA = 0.1

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


stereonet = StereoNet()
stereonet.cuda()
loadPretrain(stereonet,predModel)

print '---'
dnet = DNet()
dnet.cuda()
# loadPretrain(dnet,dnetPreModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
sceneDataset = SceneflowDataset(transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((5,30,30)),
													ToTensor(),
													normalize]))

dataloader = DataLoader(sceneDataset, batch_size=batch, shuffle=True, num_workers=8)
dataiter = iter(dataloader)

criterion1 = nn.SmoothL1Loss()
# stereoOptimizer = optim.Adam(stereonet.parameters(), lr = Lr)
# stereoOptimizer = optim.Adam([{'params':stereonet.preLoadedParams,'lr': 0},
# 								{'params':stereonet.params}], lr = stereo_lr)

# dnetOptimizer = optim.Adam(dnet.parameters(), lr = dnet_lr)

stereoOptimizer = optim.SGD(stereonet.parameters(), lr = 0, momentum = 0)
dnetOptimizer = optim.SGD(dnet.parameters(), lr = 0, momentum = 0)

one = torch.FloatTensor([1])
mone = one * -1
one = one.cuda()
mone = mone.cuda()

lossplot = []
running_loss = 0.0
running_d_loss = 0.0
running_fake_d_loss = 0.0
running_real_d_loss = 0.0
running_g_loss = 0.0

ind = 0
while True:

	ind = ind+1

	try:
		sample = dataiter.next()
	except StopIteration:
		dataiter = iter(dataloader)
		sample = dataiter.next()

	leftTensor = sample['left']
	rightTensor = sample['right']
	targetdisp = sample['disp']

	for p in dnet.parameters():
		p.requires_grad = True;

	dnetOptimizer.zero_grad()

	# train with real
	real_data_v = Variable(targetdisp).cuda()
	output_real_d = dnet(real_data_v)
	errD_real = output_real_d.mean()
	errD_real.backward(mone)

	# train with fake
	fake = stereonet(Variable(leftTensor.cuda(),requires_grad=True),Variable(rightTensor.cuda(),requires_grad=True))
	# print inputVariable.size()
	output_fake_d = dnet(fake)
	errD_fake = output_fake_d.mean()
	errD_fake.backward(one)

	# gradient_penalty = calc_gradient_penalty(dnet, real_data_v.data, fake.data)
	# gradient_penalty.backward()


	errD = errD_real - errD_fake #+ gradient_penalty
	Wasserstein_D = errD_real - errD_fake

	dnetOptimizer.step()

	# train the stereNet
	try:
		sample = dataiter.next()
	except StopIteration:
		dataiter = iter(dataloader)
		sample = dataiter.next()

	leftTensor = sample['left']
	rightTensor = sample['right']
	targetdisp = sample['disp']

	for p in dnet.parameters():
		p.requires_grad = False

	stereoOptimizer.zero_grad()

	fake = stereonet(Variable(leftTensor.cuda(),requires_grad=True),Variable(rightTensor.cuda(),requires_grad=True))
	errSup = criterion1(fake,Variable(targetdisp.cuda()))
	errSup.backward(retain_graph=True)

	output_g = dnet(fake)
	errG = output_g.mean()
	errG.backward(mone)

	# stereoOptimizer.step()




	running_loss += errSup.data[0]
	running_d_loss += errD.data[0]
	running_fake_d_loss += errD_fake.data[0]
	running_real_d_loss += -errD_real.data[0]
	running_g_loss += -errG.data[0]

	if ind % showiter == 0:    # print every 20 mini-batches
		timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
		print('%s[%d %s] stereo_loss:%.5f, g_loss:%.5f, d_loss:%.5f, D(x):%.5f  D(G):%.5f' %
		(exp_prefix[0:-1], ind, timestr, running_loss / showiter, running_g_loss/showiter, running_d_loss/showiter,
			running_real_d_loss/showiter, running_fake_d_loss/showiter))
		running_loss = 0.0
		running_d_loss = 0.0
		running_fake_d_loss = 0.0
		running_real_d_loss = 0.0
		running_g_loss = 0.0
	lossplot.append(errSup.data[0])
	# print loss.data[0]

	if (ind)%snapshot==0:
		torch.save(stereonet.state_dict(), paramName+'_'+str(ind)+'.pkl')
		torch.save(dnet.state_dict(),paramName+'_dnet_'+str(ind)+'.pkl')

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

