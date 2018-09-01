import numpy as np
import cv2
from os.path import join
from torch.utils.data import Dataset, DataLoader

class KittiDataset(Dataset):
	"""scene flow synthetic dataset. """

	def __init__(self, scale = 1, transform = None, surfix=''):
		self.scale = scale
		# self.mean = np.require([104, 117, 123], dtype=np.float32)[np.newaxis, np.newaxis,:]
		self.kittiLeftDir = '/home/wenswa/data/data_stereo_flow/training/colored_0'
		self.kittiRightDir = '/home/wenswa/data/data_stereo_flow/training/colored_1'
		self.kittiDispDir = '/home/wenswa/data/data_stereo_flow/training/disp_occ'
		self.kittiLeftDir2 = '/home/wenswa/data/data_scene_flow/training/image_2'
		self.kittiRightDir2 = '/home/wenswa/data/data_scene_flow/training/image_3'
		self.kittiDispDir2 = '/home/wenswa/data/data_scene_flow/training/disp_occ_0'

		self.N = 194 + 200
		if surfix == 'train': # seperate the last 50 for testing
			self.N = 194 + 150
		elif surfix =='test': 
			self.N = 50

		self.surfix = surfix

		self.transform = transform


	def __len__(self):
		return self.N

	def __getitem__(self, idx):
		if idx>=194:
			leftdir = self.kittiLeftDir2
			rightdir = self.kittiRightDir2
			dispdir = self.kittiDispDir2
			idx = idx - 194
		else:
			leftdir = self.kittiLeftDir
			rightdir = self.kittiRightDir
			dispdir = self.kittiDispDir

		if self.surfix == 'test': # testing case
			leftdir = self.kittiLeftDir2
			rightdir = self.kittiRightDir2
			dispdir = self.kittiDispDir2
			idx = idx + 150

		imagename = '0'*(6-len(str(idx))) + str(idx) + '_10.png'

		leftImgFile = join(leftdir, imagename)
		rightImgFile = join(rightdir, imagename)
		dispFile = join(dispdir, imagename)

		leftImg = cv2.imread(leftImgFile)
		rightImg = cv2.imread(rightImgFile)
		dispImg = cv2.imread(dispFile)
		if dispImg is not None:
			dispImg = dispImg[:,:,0]
		else:
			print '!! error reading disp image:', join(dispdir, imagename) 

		(imgHeight,imgWidth) = dispImg.shape


		# resize the images
		dispImg = cv2.resize(dispImg,(int(imgWidth/self.scale),int(imgHeight/self.scale)),interpolation=cv2.INTER_NEAREST)
		dispImg = dispImg[:,:,np.newaxis]
		dispImg = dispImg/self.scale # shrink the disparity due to the shrink of the image size

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

# normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# sceneDataset = KittiDataset(scale=0.5,transform=Compose([ RandomCrop(size=(300,720)),
# 													RandomHSV((8,60,60)),
# 													ToTensor(),
# 													normalize]),surfix='test')

# dataloader = DataLoader(sceneDataset, batch_size=1, shuffle=True, num_workers=4)
# print 'len:',len(sceneDataset)

# --- train the dataset
# for k in range(15):
# 	# print sceneDatasest[k*100]['left'].shape, sceneDataset[k*100]['right'].shape,sceneDataset[k*100]['disp'].shape
# 	sample = sceneDataset[k*30]
# 	print k*30
# 	print torch.max(sample['left']),torch.min(sample['left'])
# 	print torch.max(sample['right']),torch.min(sample['right'])
# 	print ''

# 	sampleVis = sample2img(sample,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# 	# visualize
# 	dispvis = np.tile(sampleVis['disp'],(1,1,3)).astype(np.uint8)
# 	img = np.concatenate((sampleVis['left'],sampleVis['right'],dispvis),axis=0)

# 	cv2.imshow('img',img)
# 	cv2.waitKey(0)
# ---

# --- train with the dataloader ---
# print len(sceneDataset)
# ind = 0
# for sample in dataloader:
# 	print ind
# 	ind = ind + 1
# 	print sample['left'].size(), sample['right'].size(),sample['disp'].size()
# --- 

# --- train with the iterator
# sampleiter = iter(dataloader)
# ind = 0
# while True:
# 	print ind
# 	ind = ind + 1
# 	try:
# 		sample = sampleiter.next()
# 	except StopIteration:
# 		sampleiter = iter(dataloader)
# 		sample = sampleiter.next()

# 	sampleVis = sample2img(sample,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# 	# visualize
# 	dispvis = np.tile(sampleVis['disp'],(1,1,3)).astype(np.uint8)
# 	img = np.concatenate((sampleVis['left'],sampleVis['right'],dispvis),axis=0)

# 	cv2.imshow('img',img)
# 	cv2.waitKey(0)
# ---




