
from __future__ import print_function

import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from CSN.ConvolutionalStereoNet import ConvolutionalStereoNet
from Datasets.AirsimStereoDataset import AirsimStereoDataset, Downsample, ToTensor, Normalize

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
        self.add_accumulated_value("loss2", 10)
        self.add_accumulated_value("lossLeap")
        self.add_accumulated_value("testAvg1", 10)
        self.add_accumulated_value("testAvg2", 20)
        self.add_accumulated_value("lossTest")

        # === Create a AccumulatedValuePlotter object for ploting. ===
        avNameList    = ["loss", "loss2", "lossLeap"]
        avAvgFlagList = [  True,   False,      True ]
        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "Combined", self.AV, avNameList, avAvgFlagList)\
        )

        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "loss", self.AV, ["loss"])\
        )

        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "losse", self.AV, ["loss2"], [True])\
        )

        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "lossLeap", self.AV, ["lossLeap"], [True])\
        )
        self.AVP[-1].title = "Loss Leap"

        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "testAvg1", self.AV, ["testAvg1"], [True])\
        )

        self.AVP.append(\
            WorkFlow.VisdomLinePlotter(\
                "testAvg2", self.AV, ["testAvg2"], [True])\
        )

        # === Custom member variables. ===
        self.countTrain = 0
        self.countTest  = 0

        # Cuda stuff.
        self.cudaDev = None

        # ConvolutionalStereoNet.
        self.csn        = None
        self.dataset    = None
        self.dataLoader = None

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
        self.dataset.transforms = cm

        # Dataloader.
        self.dataLoader = DataLoader( self.dataset,\
            batch_size  = self.params["torchBatchSize"],\
            shuffle     = self.params["torchShuffle"],\
            num_workers = self.params["torchNumWorkers"],\
            drop_last   = self.params["torchDropLast"] )
        
        # Optimizer.
        self.optimizer = torch.optim.Adam( \
            self.csn.parameters(), lr = params["torchOptimLearningRate"] )

        self.logger.info("Initialized.")

    # Overload the function train().
    def train(self):
        super(MyWF, self).train()

        # === Custom code. ===
        self.logger.info("Train loop #%d" % self.countTrain)

        # Test the existance of an AccumulatedValue object.
        if ( True == self.have_accumulated_value("loss") ):
            self.AV["loss"].push_back(math.sin( self.countTrain*0.1 ), self.countTrain*0.1)
        else:
            self.logger.info("Could not find \"loss\"")

        # Directly access "loss2" without existance test.
        self.AV["loss2"].push_back(math.cos( self.countTrain*0.1 ), self.countTrain*0.1)

        # lossLeap.
        if ( self.countTrain % 10 == 0 ):
            self.AV["lossLeap"].push_back(\
                math.sin( self.countTrain*0.1 + 0.25*math.pi ),\
                self.countTrain*0.1)

        # testAvg.
        self.AV["testAvg1"].push_back( 0.5, self.countTrain )

        if ( self.countTrain < 50 ):
            self.AV["testAvg2"].push_back( self.countTrain, self.countTrain )
        else:
            self.AV["testAvg2"].push_back( 50, self.countTrain )

        if ( self.countTrain % 10 == 0 ):
            self.write_accumulated_values()

        self.countTrain += 1

        # Plot accumulated values.
        self.plot_accumulated_values()

        self.logger.info("Trained.")

        time.sleep(0.05)

    # Overload the function test().
    def test(self):
        super(MyWF, self).test()

        # === Custom code. ===
        # Test the existance of an AccumulatedValue object.
        if ( True == self.have_accumulated_value("lossTest") ):
            self.AV["lossTest"].push_back(0.01, self.countTest)
        else:
            self.logger.info("Could not find \"lossTest\"")

        self.logger.info("Tested.")

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        # === Custom code. ===
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

        for i in range(100):
            wf.train()

        # Test and finalize.
        print_delimeter(title = "Test and finalize.")

        wf.test()
        wf.finalize()

        # # Show the accululated values.
        # print_delimeter(title = "Accumulated values.")
        # wf.AV["loss"].show_raw_data()

        # print_delimeter()
        # wf.AV["lossLeap"].show_raw_data()

        # print_delimeter()
        # wf.AV["lossTest"].show_raw_data()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")
