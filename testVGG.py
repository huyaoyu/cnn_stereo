
from torchvision.models.vgg import vgg16_bn
vggbn16 = vgg16_bn()
# print vggbn16
vggbn16.load_state_dict(torch.load('vgg16_bn.pth'))
# print vggbn16.state_dict()

from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

# path = '/home/wenswa/workspace/pytorch/mycode/stereo/img'
path = '/data/hdd2/wenswa/val'
imgdataset = datasets.ImageFolder(path, transforms.Compose([
		transforms.Scale(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	]))
val_loader = torch.utils.data.DataLoader(
	imgdataset,
	batch_size=1, shuffle=False,
	num_workers=1, pin_memory=True)

vggbn16.cuda()
vggbn16.eval()

import time
top3right = 0
totalnum = 0
end = time.time()
for i, (input, target) in enumerate(val_loader):
	# print 'target:',target.numpy(),imgdataset.classes[target.numpy()[0]]
	# target = target.cuda(async=True)
	input_var = torch.autograd.Variable(input.cuda(), volatile=True)
	# target_var = torch.autograd.Variable(target, volatile=True)

	# compute output
	output = vggbn16(input_var)
	# print 'output data:',output.data.cpu().topk(3)

	# print type(int(target.numpy()[0])),target.numpy()[0],type(output.data.cpu().topk(3)[1]),output.data.cpu().topk(3)[1]
	res = int(target.numpy()[0])
	top3res = output.data.cpu().topk(3)[1].view(-1)
	if res in top3res:
		top3right=top3right+1
	totalnum = totalnum+1
	if i%100==0:
		print float(top3right)/totalnum
	# loss = criterion(output, target_var)


	# measure elapsed time
	# batch_time.update(time.time() - end)
	# end = time.time()

	# print('Test: [{0}/{1}]\t'
	# 	  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
	# 	  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
	# 	  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
	# 	  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
	# 	   i, len(val_loader), batch_time=batch_time, loss=losses,
	# 	   top1=top1, top5=top5))
