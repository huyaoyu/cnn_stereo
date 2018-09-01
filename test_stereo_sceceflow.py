import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from StereoNet3 import StereoNet3 as StereoNet
# from StereoNet import StereoNet
from sceneflowDataset import SceneflowDataset,sample2img
from kittiDataset import KittiDataset
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import time
import random
np.set_printoptions(precision=4, threshold=np.nan)

# paramName = 'models/'+exp_prefix+'stereo_2'
predModel = 'models/12-3-4_stereo_gan_80000.pkl'

dataset = 'scene'

stereonet = StereoNet()
stereonet.cuda()
loadPretrain(stereonet,predModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
if dataset=='scene':
	sceneDataset = SceneflowDataset(filename = 'val.txt',transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((0,0,0)),
													ToTensor(),
													normalize]))
else:
	sceneDataset = KittiDataset(transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((0,0,0)),
													ToTensor(),
													normalize]))

dataloader = DataLoader(sceneDataset, batch_size=1, shuffle=True, num_workers=8)


# criterion = nn.SmoothL1Loss()
criterion = nn.L1Loss()
# stereoOptimizer = optim.Adam(stereonet.parameters(), lr = Lr)

stereonet.eval()

ind = 0
ImgNum = 100
losses=[0]*ImgNum
mae=0
for sample in dataloader:


	ind = ind+1

	leftTensor = sample['left']
	rightTensor = sample['right']
	targetdisp = sample['disp']
	label = targetdisp.numpy()[0,0,:,:]
	mask = targetdisp<=0 # in kitti dataset, value -1 is set to unmesured pixels
	mask2 = targetdisp>0
	validPixel = mask2.sum()

	# forward + backward + optimize
	output = stereonet(Variable(leftTensor.cuda(),volatile=True),Variable(rightTensor.cuda(),volatile=True))
	output2 = output * Variable(mask2.cuda().float(), requires_grad=False) + Variable(targetdisp.cuda(), requires_grad=False) * Variable(mask.cuda().float(), requires_grad=False) # filter for kitti spart GT
	loss = criterion(output2, Variable(targetdisp.cuda()))

	print loss.data.cpu(), validPixel, loss.data.cpu()[0]*320*640/validPixel

	# visualize
	outputDisp = output.data.cpu().numpy()
	sampleVis = sample2img(sample,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	dispvis = np.tile(sampleVis['disp'],(1,1,3)).astype(np.uint8)
	resvis = outputDisp[0,:,:,:].transpose(1,2,0)
	resvis = np.tile(resvis,(1,1,3)).astype(np.uint8)
	leftimg = sampleVis['left'].copy()
	rightimg = sampleVis['right'].copy()

	# visualize the error see why disparity differ
	(H,W) = label.shape
	for w in range(0,W,20):
		wrand = min(int(random.random()*10 + w),W-1)
		for h in range(0,H,20):
			hrand = min(int(random.random()*10 + h),H-1)
			disp = int(outputDisp[0,0,hrand,wrand])
			dispLabel = int(label[hrand,wrand])
			if dispLabel>0:
				if np.absolute(disp - dispLabel)>5:
					print '    -- ', dispLabel, '  --  ', disp
					# print 'leftimg: ',type(leftimg), leftimg.shape, leftimg.dtype
					cv2.line(leftimg,(wrand,hrand),(wrand-disp,hrand),(255,255,0),thickness=3)
					cv2.line(leftimg,(wrand,hrand),(wrand-dispLabel,hrand),(0,0,255),thickness=2)
					cv2.circle(leftimg,(wrand,hrand),1,(0,0,255),thickness=2)
					cv2.circle(rightimg,(wrand-disp,hrand),1,(255,255,0),thickness=2)
					cv2.circle(rightimg,(wrand-dispLabel,hrand),1,(0,0,255),thickness=2)


	img1 = np.concatenate((leftimg,rightimg),axis=0)
	img2 = np.concatenate((dispvis,resvis),axis=0)
	img = np.concatenate((img1,img2),axis=1)

	E = np.absolute(label - output2.data.cpu().numpy()[0,0,:,:])
	loss3p = float(np.sum(E>3))/float(np.sum(label>0))
# 	# print E
	print 'loss: ', np.sum(E>3), np.sum(label>0), loss3p*100, float(np.sum(E))/np.sum(E>0)
	losses[ind] = loss3p*100
	mae = mae + float(np.sum(E))/np.sum(E>0)
# 	# myloss = np.sum(E)
# 	# print 'myloss: ', myloss0, myloss
# 	pts = np.array([[0,0],[100,0],[100,20],[0,20]],np.int32)
# 	cv2.fillConvexPoly(img,pts,(70,30,10))
# 	cv2.putText(img,'error = {:s}'.format(str(loss*100)[0:5]),(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)
	# cv2.imwrite('img'+str(ind)+'.jpg',img)
	cv2.imshow('img',img)
	cv2.waitKey(0)

	if ind == ImgNum-1:
		break
print 'error: ',np.average(losses)
print 'MAE: ', mae/ImgNum