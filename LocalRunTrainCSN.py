
from __future__ import print_function

import argparse
import json
import math
import matplotlib.pyplot as plt
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from CSN.ConvolutionalStereoNet import ConvolutionalStereoNet
from Datasets.AirsimStereoDataset import AirsimStereoDataset, Downsample, ToTensor, Normalize
from Datasets import MiddleburyDataset

for _p in os.environ["CUSTOM_PYTHON_PATH"].split(":")[:-1]:
    sys.path.append( _p )

from workflow import WorkFlow

# ============== Constants. ======================

DEFAULT_INPUT = "inputTrain.json"

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range(int(n/2.0 + 0.5))]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

# Template for custom WorkFlow object.
class MyWF(WorkFlow.WorkFlow):
    def __init__(self, params):
        super(MyWF, self).__init__(params["workDir"], params["jobPrefix"], params["jobSuffix"])

        self.params = params

        self.verbose = params["wfVerbose"]

        # === Create the AccumulatedObjects. ===
        self.AV["loss"].avgWidth = 10
        self.add_accumulated_value("lossTest", 2)

        # === Create a AccumulatedValuePlotter object for ploting. ===
        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "loss", self.AV, \
                ["loss", "lossTest"], \
                [True, False], semiLog = True)\
        )

        # === Custom member variables. ===
        self.countTrain = 0
        self.countTest  = 0

        # Cuda stuff.
        self.cudaDev = None

        # ConvolutionalStereoNet.
        self.csn            = None
        self.dataset        = None
        self.dataLoader     = None
        self.dlIter         = None # The iterator stems from self.dataLoader.
        self.datasetTest    = None
        self.dataLoaderTest = None
        self.dlIterTest     = None # The iterator stems from self.dataLoaderTest.

        # Training variables.
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = None

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        # === Custom code. ===
        # Cuda stuff.
        self.cudaDev = torch.device(self.params["torchCudaDevice"])
        torch.cuda.set_device( self.cudaDev.index )

        # ConvolutionalStereoNet.
        self.csn = ConvolutionalStereoNet()
        self.csn.to( self.cudaDev )

        # Dataset.
        self.dataset = AirsimStereoDataset(\
            self.params["dataDir"], self.params["imageDir"], self.params["depthDir"] )
        
        # Dataset transformer.
        cm = transforms.Compose( [\
            Downsample((self.params["downSampleRatio"], self.params["downSampleRatio"])),\
            ToTensor(),\
            Normalize(self.dataset.mean, self.dataset.std)] )
        self.dataset.transform = cm

        # Dataloader.
        self.dataLoader = DataLoader( self.dataset,\
            batch_size  = self.params["torchBatchSize"],\
            shuffle     = self.params["torchShuffle"],\
            num_workers = self.params["torchNumWorkers"],\
            drop_last   = self.params["torchDropLast"] )

        self.dlIter = iter( self.dataLoader )
        
        # Optimizer.
        self.optimizer = torch.optim.Adam( \
            self.csn.parameters(), lr = params["torchOptimLearningRate"] )
        
        # Test dataset.
        self.datasetTest = MiddleburyDataset.MiddleburyDataset("/home/yyhu/expansion/OriginalData/MiddleburyDataSets/stereo/2003", "DSFiles.json")

        # Test dataset transformer.
        cmTest = transforms.Compose( [\
            # MiddleburyDataset.Crop([ 14, 1472 ]),\
            MiddleburyDataset.DownsampleCrop((320, 640)),\
            MiddleburyDataset.Downsample((2, 2)),\
            MiddleburyDataset.ToTensor(),\
            MiddleburyDataset.Normalize( self.datasetTest.mean, self.datasetTest.std )] )
        self.datasetTest.transform = cmTest

        # Dataloader.
        self.dataLoaderTest = DataLoader( self.datasetTest,\
            batch_size  = 1,\
            shuffle     = False,\
            num_workers = 1,\
            drop_last   = False )
        
        self.dlIterTest = iter( self.dataLoaderTest )

        # Load module from file.
        if ( len( self.params["loadModule"] ) != 0 ):
            self.load_modules( self.params["loadModule"] )
            self.logger.info("Module loaded from %s." % (self.params["loadModule"]))

        self.logger.info("Initialized.")

    def load_modules(self, fn):
        if ( False == self.isInitialized ):
            raise WorkFlow.WFException("Cannot load module before initialization.", "load_modules")

        self.csn.load_state_dict( torch.load( fn ) )
        

    def single_train(self, sample, md, cri, opt):
        """
        md:  The pytorch module.
        cir: The criterion.
        opt: The optimizer.
        """
        
        # Get the images and the depth files.
        image0     = sample["image0"]
        image1     = sample["image1"]
        disparity0 = sample["disparity0"]

        # Transfer data to the GPU.
        image0     = image0.to( self.cudaDev )
        image1     = image1.to( self.cudaDev )
        disparity0 = disparity0.to( self.cudaDev )

        # Clear the gradients.
        opt.zero_grad()
        
        # Forward.
        output = md( image0, image1 )
        loss   = cri( output, disparity0 )

        # Handle the loss value.
        self.AV["loss"].push_back( loss.item() )

        # Backward.
        loss.backward()
        opt.step()

    # Overload the function train().
    def train(self):
        super(MyWF, self).train()

        # === Custom code. ===
        self.logger.info("Train loop #%d" % self.countTrain)
        
        # Get new sample to train.
        try:
            sample = self.dlIter.next()
        except StopIteration as exp:
            self.dlIter = iter( self.dataLoader )
            sample = self.dlIter.next()

        # Train for single round.
        self.single_train( sample, self.csn, self.criterion, self.optimizer )

        # === Accumulated values. ===
        if ( self.countTrain % 10 == 0 ):
            self.write_accumulated_values()

        self.countTrain += 1

        # Plot accumulated values.
        self.plot_accumulated_values()

        return True

    def single_test(self, identifier, sample, md, cri):
        """
        identifier: A string identifies this test.
        md:  The pytorch module.
        """
        
        # Get the images and the depth files.
        image0     = sample["image0"]
        image1     = sample["image1"]
        disparity0 = sample["disparity0"]

        # Transfer data to the GPU.
        image0     = image0.to( self.cudaDev )
        image1     = image1.to( self.cudaDev )
        disparity0 = disparity0.to( self.cudaDev )
        
        # Forward.
        output = md( image0, image1 )
        loss   = cri( output, disparity0 )

        # Handle the loss value.
        plotX = self.countTrain - 1
        if ( plotX < 0 ):
            plotX = 0
        self.AV["lossTest"].push_back( loss.item(), plotX )

        # Save the test result.
        batchSize = output.size()[0]
        
        for i in range(batchSize):
            outDisp = output[i, 0, :, :].detach().cpu().numpy()
            gdtDisp = disparity0[i, 0, :, :].detach().cpu().numpy()

            outDisp = outDisp - outDisp.min()
            gdtDisp = gdtDisp - outDisp.min()

            outDisp = outDisp / outDisp.max()
            gdtDisp = gdtDisp / gdtDisp.max()

            # Create a matplotlib figure.
            fig = plt.figure()

            ax = plt.subplot(2, 1, 1)
            plt.tight_layout()
            ax.set_title("Ground truth")
            ax.axis("off")
            plt.imshow( gdtDisp )

            ax = plt.subplot(2, 1, 2)
            plt.tight_layout()
            ax.set_title("Prediction")
            ax.axis("off")
            plt.imshow( outDisp )

            figName = "%s_%02d" % (identifier, i)
            figName = self.compose_file_name(figName, "png")
            plt.savefig(figName)

            plt.close(fig)

    # Overload the function test().
    def test(self):
        super(MyWF, self).test()

        # === Custom code. ===

        # Get new sample to test.
        try:
            sample = self.dlIterTest.next()
        except StopIteration as exp:
            self.dlIterTest = iter( self.dataLoaderTest )
            sample = self.dlIterTest.next()

        # Single test.
        identifier = "test_%d" % (self.countTrain - 1)
        self.single_test(identifier, sample, self.csn, self.criterion)

        # Plot accumulated values.
        self.plot_accumulated_values()

        self.logger.info("Tested.")

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        # === Custom code. ===

        # Save the CNN into file system.
        fn = self.compose_file_name("csn", "pmd")
        torch.save( self.csn.state_dict(), fn )

        self.logger.info("Finalized.")

if __name__ == "__main__":
    # Arguments.
    parser = argparse.ArgumentParser(description="Train a CNN with workflow.")

    parser.add_argument("--input", help = "The filename of the input JSON file.", default = DEFAULT_INPUT)
    # parser.add_argument("--voiddiam_m",\
    #     help = "Overwrite voiddiam_m in the input JSON file.",\
    #     default = -1.0, type = float)
    # parser.add_argument("--write", help = "Write multiple arrays to file system.", action = "store_true", default = False)

    args = parser.parse_args()

    # Load the input json file.
    fp = open( args.input, "r" )
    params = json.load(fp)
    fp.close()
    
    print("%s" % (params["jobName"]))

    print_delimeter(title = "Before initialization." )

    try:
        # Instantiate an object for MyWF.
        wf = MyWF(params)

        # Initialization.
        print_delimeter(title = "Initialize.")
        wf.initialize()

        # Training loop.
        print_delimeter(title = "Loop.")

        testInterval = params["testInterval"]

        for i in range( params["trainLoops"] ):
            if ( i % testInterval == 0 ):
                wf.test()
            
            if ( False == wf.train() ):
                break

        # Test and finalize.
        print_delimeter(title = "Test and finalize.")

        wf.test()
        wf.finalize()

    except WorkFlow.SigIntException as e:
        print( e.describe() )
        print( "Interrupted by SIGINT. Entering finalize()." )
        wf.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")
