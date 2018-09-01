import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from StereoNet import StereoNet
from DNet import DNet
from sceneflowDataset import SceneflowDataset
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(precision=4, threshold=np.nan)

exp_prefix = '4-3_'
Lr = 1e-4
batch = 1
trainstep = 100000
showiter = 10
snapshot = 1000
paramName = 'models/'+exp_prefix+'dnet'
predModel = 'models/1-4_stereo_2_45000.pkl'
dnetPreModel = 'models/4-2_dnet_2000.pkl'

stereonet = StereoNet()
stereonet.cuda()
loadPretrain(stereonet,predModel)

print '---'
dnet = DNet()
dnet.cuda()
loadPretrain(dnet,dnetPreModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
sceneDataset = SceneflowDataset(transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((10,100,100)),
													ToTensor(),
													normalize]))
dataloader = DataLoader(sceneDataset, batch_size=batch, shuffle=True, num_workers=8)


criterion = nn.BCELoss()
# stereoOptimizer = optim.Adam(stereonet.parameters(), lr = Lr)
dnetOptimizer = optim.Adam(dnet.parameters(), lr = Lr)

label = torch.FloatTensor(batch)
input = torch.FloatTensor(batch,320,640)
real_label = 1
fake_label = 0
label = Variable(label)
input = Variable(input)
label = label.cuda()

lossplot = []
running_loss = 0.0

ind = 0
while True:

	for sample in dataloader:

		ind = ind+1

		leftTensor = sample['left']
		rightTensor = sample['right']
		targetdisp = sample['disp']

		dnetOptimizer.zero_grad()

		inputVariable = Variable(torch.cat((leftTensor,targetdisp),dim=1),requires_grad=True)
		# print inputVariable.size()
		output = dnet(inputVariable.cuda())
		label.data.resize_(batch).fill_(real_label)
		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.data.mean()

		# train with fake
		fake = stereonet(Variable(leftTensor.cuda(),requires_grad=False),Variable(rightTensor.cuda(),requires_grad=False))
		label.data.fill_(fake_label)
		inputVariable = Variable(torch.cat((leftTensor.cuda(),fake.detach().data),dim=1),requires_grad=True)
		# print inputVariable.size()
		output = dnet(inputVariable)
		errD_fake = criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.data.mean()
		errD = errD_real + errD_fake
		dnetOptimizer.step()


		running_loss += errD.data[0]
		if ind % showiter == 0:    # print every 20 mini-batches
			timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
			print('[%d %s] loss: %.5f lr: %f %.5f %.5f' %
			(ind, timestr, running_loss / showiter, dnetOptimizer.param_groups[0]['lr'],D_x,D_G_z1))
			running_loss = 0.0
		lossplot.append(errD.data[0])
		# print loss.data[0]


		if (ind)%snapshot==0:
			torch.save(dnet.state_dict(), paramName+'_'+str(ind)+'.pkl')

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

