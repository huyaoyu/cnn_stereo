import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class StereoNet3(nn.Module):
	"""
	In the second version, there are poolings after 3x3 conv in the encoder-decoder, 
	which may cause grid effect. 
	Change conv to 4x4 with stride 2.
	"""
	def __init__(self,version=0):
		super(StereoNet3, self).__init__()
		self.version = version
		# feature extraction layers
		self.conv1_1 = nn.Conv2d(3,64, kernel_size=3, padding=1)
		self.bn1_1 = nn.BatchNorm2d(64)
		self.conv1_2 = nn.Conv2d(64,64, kernel_size=3, padding=1)
		self.bn1_2 = nn.BatchNorm2d(64)

		self.conv2_1 = nn.Conv2d(64,128, kernel_size=3, padding=1)
		self.bn2_1 = nn.BatchNorm2d(128)
		self.conv2_2 = nn.Conv2d(128,128, kernel_size=3, padding=1)
		self.bn2_2 = nn.BatchNorm2d(128)
		self.upscale2 = nn.UpsamplingBilinear2d(scale_factor=2)
		self.conv2s = nn.Conv2d(128,64, kernel_size=1, padding=0)

		self.conv3_1 = nn.Conv2d(128,256, kernel_size=3, padding=1)
		self.bn3_1 = nn.BatchNorm2d(256)
		self.conv3_2 = nn.Conv2d(256,256, kernel_size=3, padding=1)
		self.bn3_2 = nn.BatchNorm2d(256)
		self.conv3_3 = nn.Conv2d(256,256, kernel_size=3, padding=1)
		self.bn3_3 = nn.BatchNorm2d(256)
		self.upscale3 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.conv3s = nn.Conv2d(256,64, kernel_size=1, padding=0)

		self.pretrainMap={'conv1_1':0, 'bn1_1':1, 'conv1_2':3, 'bn1_2':4, 
			'conv2_1':7, 'bn2_1':8, 'conv2_2':10, 'bn2_2':11, 
			'conv3_1':14, 'bn3_1':15, 'conv3_2':17, 'bn3_2':18, 'conv3_3':20, 'bn3_3':21}

		# depth regression layers
		self.conv_c0 = nn.Conv2d(390,128, kernel_size=1,stride=1, padding=0)
		self.conv_c1 = nn.Conv2d(128,128, kernel_size=4,stride=2, padding=1)
		self.conv_c2 = nn.Conv2d(128,128, kernel_size=4,stride=2, padding=1)
		self.conv_c3 = nn.Conv2d(128,256, kernel_size=4,stride=2, padding=1)
		self.conv_c4 = nn.Conv2d(256,256, kernel_size=4,stride=2, padding=1)
		self.conv_c5 = nn.Conv2d(256,512, kernel_size=4,stride=2, padding=1)
		self.conv_c6 = nn.Conv2d(512,512, kernel_size=4,stride=2, padding=1)
		self.deconv_c7 = nn.ConvTranspose2d(512, 512,kernel_size=4,stride=2,padding=1)
		self.deconv_c8 = nn.ConvTranspose2d(1024, 512,kernel_size=4,stride=2,padding=1)
		self.deconv_c9 = nn.ConvTranspose2d(768, 256,kernel_size=4,stride=2,padding=1)
		self.deconv_c10 = nn.ConvTranspose2d(512, 256,kernel_size=4,stride=2,padding=1)
		self.deconv_c11 = nn.ConvTranspose2d(384, 128,kernel_size=4,stride=2,padding=1)
		self.deconv_c12 = nn.ConvTranspose2d(256, 128,kernel_size=4,stride=2,padding=1)
		self.conv_c13 = nn.Conv2d(256, 64,kernel_size=1,padding=0)
		self.conv_c14 = nn.Conv2d(64, 1,kernel_size=1,padding=0)

		self._initialize_weights()
		# load conv1-conv3 from the pretrained VGG16_bn network
		self.loadPretrainedVGG('vgg16_bn.pth')


	def featureExtract(self,x):
		# feature extraction layers
		out = self.conv1_1(x)
		out = self.bn1_1(out)
		out = F.relu(out, inplace=True)
		out = self.conv1_2(out)
		cat1 = self.bn1_2(out)
		out = F.max_pool2d(F.relu(cat1, inplace=False), kernel_size=2)

		out = self.conv2_1(out)
		out = self.bn2_1(out)
		out = F.relu(out, inplace=True)
		out = self.conv2_2(out)
		cat2 = self.bn2_2(out)
		out = F.max_pool2d(F.relu(cat2, inplace=False), kernel_size=2)
		cat2 = self.upscale2(cat2)
		cat2 = self.conv2s(cat2)

		out = self.conv3_1(out)
		out = self.bn3_1(out)
		out = F.relu(out, inplace=True)
		out = self.conv3_2(out)
		out = self.bn3_2(out)
		out = F.relu(out, inplace=True)
		out = self.conv3_3(out)
		cat3 = self.bn3_3(out)
		cat3 = self.upscale3(cat3)
		cat3 = self.conv3s(cat3)

		x = torch.cat((x,cat1,cat2,cat3),dim=1)

		return x


	def forward(self, x, y):
		left = self.featureExtract(x)
		right = self.featureExtract(y)

		x = torch.cat((left,right),dim=1)

		# depth regression layers
		x = self.conv_c0(x)
		cat0 = F.relu(x, inplace=True)
		x = self.conv_c1(x)
		cat1 = F.relu(x, inplace=True)
		x = self.conv_c2(x)
		cat2 = F.relu(x, inplace=True)
		x = self.conv_c3(x)
		cat3 = F.relu(x, inplace=True)
		x = self.conv_c4(x)
		cat4 = F.relu(x, inplace=True)
		x = self.conv_c5(x)
		cat5 = F.relu(x, inplace=True)
		x = self.conv_c6(x)
		x = F.relu(x, inplace=True)

		x = self.deconv_c7(x)
		x = F.relu(x, inplace=True)
		x = torch.cat((x,cat5),dim=1)
		x = self.deconv_c8(x)
		x = F.relu(x, inplace=True)
		x = torch.cat((x,cat4),dim=1)
		x = self.deconv_c9(x)
		x = F.relu(x, inplace=True)
		x = torch.cat((x,cat3),dim=1)
		x = self.deconv_c10(x)
		x = F.relu(x, inplace=True)
		x = torch.cat((x,cat2),dim=1)
		x = self.deconv_c11(x)
		x = F.relu(x, inplace=True)
		x = torch.cat((x,cat1),dim=1)
		x = self.deconv_c12(x)
		x = F.relu(x, inplace=True)
		x = torch.cat((x,cat0),dim=1)

		x = self.conv_c13(x)
		x = F.relu(x, inplace=True)
		x = self.conv_c14(x)
		x = F.relu(x, inplace=True)

		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	# return a generator for pre-loaded weights for adjusting the learning rate
	def preLoadedParamsGen(self,loadDict):
		for name,param in self.named_parameters():
			if name in loadDict.keys():
				yield param
	def paramsGen(self,loadDict):
		for name,param in self.named_parameters():
			if name not in loadDict.keys():
				yield param


	def loadPretrainedVGG(self, preTrainModel):
		preTrainDict = torch.load(preTrainModel)
		model_dict = self.state_dict()
		loadDict = {}
		for k,v in model_dict.items():
			keys = k.split('.')
			# print keys,keys[0]
			if keys[0] in self.pretrainMap: # compansate for naming bug
				k2 = 'features.'+str(self.pretrainMap[keys[0]])+'.'+keys[-1]
				loadDict[k]=preTrainDict[k2]
				print '  Load pretrained layer: ',k2
		model_dict.update(loadDict)
		self.load_state_dict(model_dict)

		self.preLoadedParams = self.preLoadedParamsGen(loadDict)
		self.params = self.paramsGen(loadDict)
		# for param in self.preLoadedParams:
		# 	param.requires_grad = False # fix the weights for the pre-loaded layers
		# 	# print param.requires_grad
 
# from torch.autograd import Variable
# stereonet = StereoNet3()
# stereonet.cuda()
# print stereonet
# import numpy as np
# import matplotlib.pyplot as plt
# np.set_printoptions(precision=4, threshold=np.nan)
# imsize = 320
# x, y = np.ogrid[:imsize, :imsize]
# # print x, y, (x+y)
# img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(imsize + imsize)
# img = img.astype(np.float32)
# print img.dtype

# imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
# imgTensor = torch.from_numpy(imgInput)
# z = stereonet(Variable(imgTensor.cuda(),requires_grad=False),Variable(imgTensor.cuda(),requires_grad=False))
# print z.data.cpu().numpy().shape
# # print z.data.numpy()

# # for name,param in stereonet.named_parameters():
# # 	print name,param.requires_grad

