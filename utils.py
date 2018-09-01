import torch

def loadPretrain(model, preTrainModel):
	preTrainDict = torch.load(preTrainModel)
	model_dict = model.state_dict()
	print 'preTrainDict:',preTrainDict.keys()
	print 'modelDict:',model_dict.keys()
	preTrainDict = {k:v for k,v in preTrainDict.items() if k in model_dict}
	for item in preTrainDict:
		print '  Load pretrained layer: ',item
	model_dict.update(preTrainDict)
	# for item in model_dict:
	# 	print '  Model layer: ',item
	model.load_state_dict(model_dict)
	return model

