import numpy as np
import cv2
from pfm import readPFM
from os.path import join
from torch.utils.data import Dataset, DataLoader

class SceneflowDataset(Dataset):
	"""scene flow synthetic dataset. """

	def __init__(self, filename = 'train.txt', scale = 1, transform = None):
		self.scale = scale
		# self.mean = np.require([104, 117, 123], dtype=np.float32)[np.newaxis, np.newaxis,:]

		with open(filename,'r') as f:
			self.lines = f.readlines()

		self.N = len(self.lines)

		self.transform = transform


	def __len__(self):
		return self.N

	def __getitem__(self, idx):
		imgnames = self.lines[idx].split(' ')
		dispImg, scale0 = readPFM(imgnames[2].strip())
		(imgHeight,imgWidth) = dispImg.shape
		# resize the images
		dispImg = cv2.resize(dispImg,(int(imgWidth/self.scale),int(imgHeight/self.scale)))
		dispImg = dispImg[:,:,np.newaxis]
		dispImg = dispImg/self.scale # shrink the disparity due to the shrink of the image size

		# read two image files
		leftImg = cv2.imread(imgnames[0].strip())
		rightImg = cv2.imread(imgnames[1].strip())
		leftImg = cv2.resize(leftImg,(int(imgWidth/self.scale),int(imgHeight/self.scale)))
		rightImg = cv2.resize(rightImg,(int(imgWidth/self.scale),int(imgHeight/self.scale)))
		# leftImg = leftImg - self.mean # mean substraction
		# rightImg = rightImg - self.mean

		sample = {'left': leftImg, 'right': rightImg, 'disp': dispImg}

		if self.transform:
			sample = self.transform(sample)

		return sample


# from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types


class Compose(object):
	"""Composes several transforms together.

	Args:
		transforms (List[Transform]): list of transforms to compose.

	Example:
		>>> transforms.Compose([
		>>>     transforms.CenterCrop(10),
		>>>     transforms.ToTensor(),
		>>> ])
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img):
		for t in self.transforms:
			img = t(img)
		return img

class ToTensor(object):
	def __call__(self, sample):
		leftImg, rightImg, dispImg = sample['left'], sample['right'], sample['disp']
		leftImg = leftImg.transpose(2,0,1)/float(255)
		rightImg = rightImg.transpose(2,0,1)/float(255)
		dispImg = dispImg.transpose(2,0,1)
		return {'left': torch.from_numpy(leftImg).float(), 'right': torch.from_numpy(rightImg).float(), 'disp': torch.from_numpy(dispImg).float()}


class Normalize(object):
	"""Given mean: (R, G, B) and std: (R, G, B),
	will normalize each channel of the torch.*Tensor, i.e.
	channel = (channel - mean) / std
	this should be called after ToTensor
	"""

	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, sample):
		leftImg, rightImg = sample['left'], sample['right']
		for t, m, s in zip(leftImg, self.mean, self.std):
			t.sub_(m).div_(s)
		for t, m, s in zip(rightImg, self.mean, self.std):
			t.sub_(m).div_(s)
		return {'left': leftImg, 'right': rightImg, 'disp': sample['disp']}


# class Scale(object):
# 	"""Rescales the input PIL.Image to the given 'size'.
# 	'size' will be the size of the smaller edge.
# 	For example, if height > width, then image will be
# 	rescaled to (size * height / width, size)
# 	size: size of the smaller edge
# 	interpolation: Default: PIL.Image.BILINEAR
# 	"""

# 	def __init__(self, size, interpolation=Image.BILINEAR):
# 		self.size = size
# 		self.interpolation = interpolation

# 	def __call__(self, img):
# 		w, h = img.size
# 		if (w <= h and w == self.size) or (h <= w and h == self.size):
# 			return img
# 		if w < h:
# 			ow = self.size
# 			oh = int(self.size * h / w)
# 			return img.resize((ow, oh), self.interpolation)
# 		else:
# 			oh = self.size
# 			ow = int(self.size * w / h)
# 			return img.resize((ow, oh), self.interpolation)
# 		# dispImg = cv2.resize(dispImg,(int(imgWidth/self.scale),int(imgHeight/self.scale)))
# 		# dispImg = dispImg[np.newaxis,:,:]
# 		# dispImg = dispImg/self.scale # shrink the disparity due to the shrink of the image size

# 		# # read two image files
# 		# leftImg = cv2.imread(imgnames[0].strip())
# 		# rightImg = cv2.imread(imgnames[1].strip())
# 		# leftImg = cv2.resize(leftImg,(int(imgWidth/self.scale),int(imgHeight/self.scale)))
# 		# rightImg = cv2.resize(rightImg,(int(imgWidth/self.scale),int(imgHeight/self.scale)))


class RandomCrop(object):
	"""Crops the given imgage(in numpy format) at a random location to have a region of
	the given size. size can be a tuple (target_height, target_width)
	or an integer, in which case the target will be of a square shape (size, size)
	"""

	def __init__(self, size):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size

	def __call__(self, sample):
		leftImg, rightImg, dispImg = sample['left'], sample['right'], sample['disp']

		(h, w, c) = leftImg.shape
		th, tw = self.size
		if w == tw and h == th:
			return {'left': leftImg, 'right': rightImg, 'disp': dispImg}

		x1 = random.randint(0, w - tw)
		y1 = random.randint(0, h - th)

		leftImg = leftImg[y1:y1+th,x1:x1+tw,:]
		rightImg = rightImg[y1:y1+th,x1:x1+tw,:]
		dispImg = dispImg[y1:y1+th,x1:x1+tw,:]

		# print leftImg.shape,rightImg.shape,dispImg.shape

		return {'left': leftImg, 'right': rightImg, 'disp': dispImg}

class RandomHSV(object):
	"""
	Change the image in HSV space
	"""

	def __init__(self,HSVscale=(6,30,30)):
		self.Hscale, self.Sscale, self.Vscale = HSVscale


	def __call__(self, sample):
		leftImg, rightImg, dispImg = sample['left'], sample['right'], sample['disp']

		leftHSV = cv2.cvtColor(leftImg, cv2.COLOR_BGR2HSV)
		rightHSV = cv2.cvtColor(rightImg, cv2.COLOR_BGR2HSV)
		# change HSV
		h = random.random()*2-1
		s = random.random()*2-1
		v = random.random()*2-1
		leftHSV[:,:,0] = np.clip(leftHSV[:,:,0]+self.Hscale*h,0,255)
		leftHSV[:,:,1] = np.clip(leftHSV[:,:,1]+self.Sscale*s,0,255)
		leftHSV[:,:,2] = np.clip(leftHSV[:,:,2]+self.Vscale*v,0,255)
		rightHSV[:,:,0] = np.clip(rightHSV[:,:,0]+self.Hscale*h,0,255)
		rightHSV[:,:,1] = np.clip(rightHSV[:,:,1]+self.Sscale*s,0,255)
		rightHSV[:,:,2] = np.clip(rightHSV[:,:,2]+self.Vscale*v,0,255)

		leftImg = cv2.cvtColor(leftHSV,cv2.COLOR_HSV2BGR)
		rightImg = cv2.cvtColor(rightHSV,cv2.COLOR_HSV2BGR)

		return {'left': leftImg, 'right': rightImg, 'disp': dispImg}

def sample2img(sample,mean,std):
	"""
	convert dict of tensor to dict of numpy array, for visualization
	"""
	leftImg, rightImg, dispImg = sample['left'], sample['right'], sample['disp']
	if len(leftImg.size())==4:
		leftImg = leftImg[0,:,:,:]
		rightImg = rightImg[0,:,:,:]
		dispImg = dispImg[0,:,:,:]
	leftImg = tensor2img(leftImg, mean, std)
	rightImg = tensor2img(rightImg,mean,std)

	dispImg = dispImg.numpy().transpose(1,2,0)

	return {'left':leftImg, 'right': rightImg, 'disp':dispImg}

def tensor2img(tensImg,mean,std):
	"""
	convert a tensor a numpy array, for visualization
	"""
	# undo normalize
	for t, m, s in zip(tensImg, mean, std):
		t.mul_(s).add_(m)
	# undo transpose
	tensImg = (tensImg.numpy().transpose(1,2,0)*float(255)).astype(np.uint8)
	return tensImg

# test 

# from torch.utils.data import Dataset, DataLoader
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import numpy as np
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# # path = '/home/wenswa/workspace/pytorch/mycode/stereo/img'
# path = '/data/hdd2/wenswa/val'
# imgdataset = datasets.ImageFolder(path, transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ]))
# val_loader = torch.utils.data.DataLoader(
#     imgdataset,
#     batch_size=1, shuffle=False,
#     num_workers=1, pin_memory=True)




# normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# sceneDataset = SceneflowDataset(transform=Compose([ RandomCrop(size=(360,720)),
# 													RandomHSV((10,100,100)),
# 													ToTensor(),
# 													normalize]))

# dataloader = DataLoader(sceneDataset, batch_size=4, shuffle=True, num_workers=4)

# for k in range(20):
# 	# print sceneDatasest[k*100]['left'].shape, sceneDataset[k*100]['right'].shape,sceneDataset[k*100]['disp'].shape
# 	sample = sceneDataset[k*1000]
# 	print torch.max(sample['left']),torch.min(sample['left'])
# 	print torch.max(sample['right']),torch.min(sample['right'])
# 	print ''

# 	sampleVis = sample2img(sample,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# 	# visualize
# 	dispvis = np.tile(sampleVis['disp'],(1,1,3)).astype(np.uint8)
# 	img = np.concatenate((sampleVis['left'],sampleVis['right'],dispvis),axis=0)

# 	cv2.imshow('img',img)
# 	cv2.waitKey(0)
# # print len(sceneDataset)
# # for sample in dataloader:
# # 	print sample['left'].size(), sample['right'].size(),sample['disp'].size()





