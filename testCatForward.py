import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from StereoNet2 import StereoNet2
from sceneflowDataset import SceneflowDataset,sample2img
from sceneflowDataset import RandomCrop, RandomHSV, ToTensor, Normalize, Compose
from utils import loadPretrain
import numpy as np
import time
np.set_printoptions(precision=4, threshold=np.nan)

snapshot = 5000
# paramName = 'models/'+exp_prefix+'stereo_2'
predModel = 'models/7-1-2_stereo_gan_85000.pkl'

stereonet = StereoNet2()
stereonet.cuda()
loadPretrain(stereonet,predModel)

normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
sceneDataset = SceneflowDataset(filename = 'val.txt',transform=Compose([ RandomCrop(size=(320,640)),
													RandomHSV((10,100,100)),
													ToTensor(),
													normalize]))
dataloader = DataLoader(sceneDataset, batch_size=1, shuffle=True, num_workers=8)

stereonet.eval()


for sample in dataloader:


	leftTensor = sample['left']
	rightTensor = sample['right']
	targetdisp = sample['disp']

	featureleft = stereonet.featureExtract(Variable(leftTensor.cuda()))
	featureleft = featureleft.cpu().data.numpy()
	print featureleft.shape
	print featureleft[0,0,:,:]
	print '---'
	print featureleft[0,4,:,:]

	break

	# # forward + backward + optimize
	# output = stereonet(Variable(leftTensor.cuda(),volatile=True),Variable(rightTensor.cuda(),volatile=True))


# # test if the cat in forward function works
# class StereoNet(nn.Module):

# 	def __init__(self,version=0):
# 		super(StereoNet, self).__init__()
# 		# feature extraction layers
# 		self.conv1 = nn.Conv2d(1,3, kernel_size=3, padding=1)
# 		self.conv2 = nn.Conv2d(3,6, kernel_size=3, padding=1)
# 		self.conv2 = nn.Conv2d(6,8, kernel_size=3, padding=1)

# torch.manual_seed(37)
# x = torch.rand(1,1,9,9)-0.5
# x = Variable(x)
# print x
# y = F.relu(x,inplace=True)
# print '---'
# print x
# print y
# x.data[0,0,0,0]=100
# print '---'
# print x
# print y


# z = F.threshold(x,0.2,-1.0,inplace=True)
# print '---'
# print x
# print y
# print z
# # print z