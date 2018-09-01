import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from torch.logger import Logger
from StereoNet3 import StereoNet3 as StereoNet
from DNet25 import DNet25 as DNet
from DNet25 import DNet252 as DNet2
from sceneflowDataset import SceneflowDataset, sample2img
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(precision=4, threshold=np.nan)

batch = 1
predModel = 'models/12-3-4_stereo_gan_90000.pkl'
dnetPreModel = 'models/12-3-4_stereo_gan_dnet_90000.pkl'
testNum = 100

stereonet = StereoNet()
stereonet.cuda()
loadPretrain(stereonet,predModel)

print '---'
dnet = DNet()
dnet.cuda()
loadPretrain(dnet,dnetPreModel)

dnet2 = DNet2()
dnet2.cuda()
loadPretrain(dnet2,dnetPreModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
sceneDataset = SceneflowDataset(filename = 'val.txt',transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((7,37,37)),
													ToTensor(),
													normalize]))

dataloader = DataLoader(sceneDataset, batch_size=batch, shuffle=True, num_workers=2)


criterion1 = nn.SmoothL1Loss()
criterion2 = nn.BCELoss()

label = torch.FloatTensor(batch)
real_label = 1
fake_label = 0
label = Variable(label)
label = label.cuda()

ind = 0
while True:

	for sample in dataloader:

		ind = ind+1

		leftTensor = sample['left']
		rightTensor = sample['right']
		targetdisp = sample['disp']

		# train with real
		inputVariable = Variable(targetdisp,requires_grad=False)
		output = dnet(inputVariable.cuda())
		label.data.resize_(batch).fill_(real_label)
		errD_real = criterion2(output, label)
		D_x = output.data.mean()

		# train with fake
		fake = stereonet(Variable(leftTensor.cuda(),requires_grad=False),Variable(rightTensor.cuda(),requires_grad=False))
		label.data.fill_(fake_label)
		output = dnet(fake.detach())
		errD_fake = criterion2(output, label)
		D_G_z1 = output.data.mean()

		errSup = criterion1(fake,Variable(targetdisp.cuda()))

		print 'fake score: %.4f, real score: %.4f, stereo score: %.4f' % (D_G_z1, D_x, errSup.data[0])



		sampleVis = sample2img(sample,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		dispvis = np.tile(sampleVis['disp'],(1,1,3)).astype(np.uint8)
		resvis = fake[0,:,:,:].data.cpu().numpy().transpose(1,2,0)
		resvis = np.tile(resvis,(1,1,3)).astype(np.uint8)
		leftimg = sampleVis['left'].copy()
		rightimg = sampleVis['right'].copy()

		# analyze the discriminator's feature
		real_feature = dnet2(inputVariable.cuda())
		real_feature = real_feature.data.cpu()[0,:,:,:]
		real_feature = real_feature.mean(dim=0)
		print real_feature
		fake_feature = dnet2(fake.detach())
		fake_feature = fake_feature.data.cpu()[0,:,:,:]
		fake_feature = fake_feature.mean(dim=0)
		print fake_feature

		diff = (real_feature - fake_feature).numpy()
		print real_feature - fake_feature
		(H,W) = diff.shape
		fake_feature = fake_feature.numpy()
		real_feature = real_feature.numpy()
		for h in range(H):
			for w in range(W):
				if fake_feature[h,w]>0.2:
					color = min(255,int(255*(fake_feature[h,w]-0.2)*5)+100)
					cv2.circle(resvis,(w*64+32,h*64+32),32,(70,70,color),thickness=2)

				if fake_feature[h,w]<-0.2:
					color = min(255,int(255*(-0.2 - fake_feature[h,w])*5)+100)
					cv2.circle(resvis,(w*64+32,h*64+32),32,(color,70,70),thickness=2)

				if real_feature[h,w]>0.2:
					color = min(255,int(255*(real_feature[h,w]-0.2)*5)+100)
					cv2.circle(dispvis,(w*64+32,h*64+32),32,(70,70,color),thickness=2)

				if real_feature[h,w]<-0.2:
					color = min(255,int(255*(-0.2 - real_feature[h,w])*5)+100)
					cv2.circle(dispvis,(w*64+32,h*64+32),32,(color,70,70),thickness=2)
				# if diff[h,w]>0.02:
				# 	color = min(255,int(255*(diff[h,w]-0.02)*10)+100)
				# 	print color
				# 	cv2.circle(resvis,(w*64+32,h*64+32),32,(70,70,color),thickness=2)

				# if diff[h,w]<-0.02:
				# 	color = min(255,int(255*(-0.02 - diff[h,w])*10)+100)
				# 	print color
				# 	cv2.circle(resvis,(w*64+32,h*64+32),32,(color,70,70),thickness=2)


		# diffvis = (diffvis-np.min(diffvis))/(np.max(diffvis)-np.min(diffvis))
		# diffvis = cv2.resize(diffvis,(100,50),interpolation=cv2.INTER_NEAREST)
		# cv2.imshow('diff',diffvis)

		img1 = np.concatenate((leftimg,rightimg),axis=0)
		img2 = np.concatenate((dispvis,resvis),axis=0)
		img = np.concatenate((img1,img2),axis=1)

		# cv2.imshow('img',img)
		# cv2.waitKey(0)
		cv2.imwrite('disc_test_fake/img'+str(ind)+'.jpg',img)


		if ind>testNum:
			break

	if ind>testNum:
		break


