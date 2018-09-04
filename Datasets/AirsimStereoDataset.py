
from __future__ import print_function

import cv2
import glob
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def find_filenames(d, fnPattern):
    """
    Find all the filenames in directory d. A ascending sort is applied by default.

    d: The directory.
    fnPattern: The file pattern like "*.png".
    return: A list contains the strings of the sortted file names.
    """

    # Compose the search pattern.
    s = d + "/" + fnPattern

    fnList = glob.glob(s)
    fnList.sort()

    return fnList

class Downsample(object):
    def __init__(self, newSize):
        assert isinstance( newSize, (int, tuple) )

        self.newSize = newSize

        if isinstance( self.newSize, int ):
            if ( self.newSize <= 0 ):
                raise Exception("The downsample size must be a positive integer.")
            
            self.ratio = [ 1.0 / newSize, 1.0 / newSize ]
        else:
            if ( self.newSize[0] <= 0 or self.newSize[1] <= 0 ):
                raise Exception("The downsample size must be two positive integers if using tuple.")
            
            if ( not isinstance( self.newSize[0], int ) or not isinstance( self.newSize[1], int ) ):
                raise Exception("The downsample size must be two integers.")
            
            self.ratio = [ 1.0 / newSize[0], 1.0 / newSize[1] ]

    def __call__(self, sample):
        # Retreive the images and the depths.
        image0 = sample["image0"]
        image1 = sample["image1"]
        depth0 = sample["depth0"]
        depth1 = sample["depth1"]

        # Downsample the iamges.
        image0 = cv2.resize( image0, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        image1 = cv2.resize( image1, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        depth0 = cv2.resize( depth0, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )
        depth1 = cv2.resize( depth1, (0, 0), fx = self.ratio[0], fy = self.ratio[1], interpolation = cv2.INTER_NEAREST )

        return { "image0": image0, "image1": image1, "depth0": depth0, "depth1": depth1 }

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # Retreive the images and the depths.
        image0 = torch.from_numpy( sample["image0"].transpose((2, 0, 1)) )
        image1 = torch.from_numpy( sample["image1"].transpose((2, 0, 1)) )

        depthShape = sample["depth0"].shape

        depth0 = torch.from_numpy( sample["depth0"].reshape((1, depthShape[0], depthShape[1])) )
        depth1 = torch.from_numpy( sample["depth1"].reshape((1, depthShape[0], depthShape[1])) )

        return { "image0": image0, "image1": image1, "depth0": depth0, "depth1": depth1 }

class AirsimStereoDataset(Dataset):
    def __init__(self, rootDir, imageDir, depthDir, imageExt = "png", depthExt = "npy", imageSuffix = "_rgb", depthSuffix = "_depth", transform = None):
        self.rootDir     = rootDir
        self.imageDir    = imageDir
        self.depthDir    = depthDir
        self.imageExt    = imageExt    # File extension of image file.
        self.depthExt    = depthExt    # File extension of depth file.
        self.imageSuffix = imageSuffix
        self.depthSuffix = depthSuffix
        self.transform   = transform

        # Find all the file names.
        self.imageFiles = find_filenames( \
            self.rootDir + "/" + self.imageDir,\
            "*." + self.imageExt )
        self.depthFiles = find_filenames( \
            self.rootDir + "/" + self.depthDir,\
            "*." + self.depthExt )

        # Check the number of files.
        self.nImageFiles = len( self.imageFiles )
        self.nDepthFiles = len( self.depthFiles )
        if ( self.nImageFiles != self.nDepthFiles ):
            raise Exception("The numbers of files found are not consistent with each other. nImageFiles = {}, nDepthFiles = {}".format(self.nImageFiles, self.nDepthFiles))
        
        if ( 0 == self.nImageFiles ):
            raise Exception("No files found.")

        if ( self.nImageFiles & 0x01 == 0x01 ):
            raise Exception("There must be even number of files.")

        if ( False == self.check_filenames( \
            self.imageFiles, self.depthFiles, self.imageSuffix, self.depthSuffix, self.imageExt, self.depthExt ) ):
            raise Exception("File names are not consistent.")
    
    def check_filenames(self, fnList0, fnList1, suffix0, suffix1, ext0, ext1):
        """
        This function checks the filenames in fnList0 and fnList1 one by one. If all the
        filenames are consistent, then this function returns True. It returns False
        otherwise. When camparing the filenames, the suffix of each filename will be ignored.

        It is assumed that the lengths of fnList0 and fnList1 are the same.
        """

        nFiles   = len(fnList0)
        nSuffix0 = len(suffix0)
        nSuffix1 = len(suffix1)
        nExt0    = len(ext0)
        nExt1    = len(ext1)

        for i in range(nFiles):
            f0 = os.path.basename( fnList0[i] )[:-nSuffix0-nExt0-1]
            f1 = os.path.basename( fnList1[i] )[:-nSuffix1-nExt1-1]

            if ( f0 != f1 ):
                return False
        
        return True

    def __len__(self):
        return (int)( self.nImageFiles / 2 )

    def __getitem__(self, idx):
        """
        A dictionary is returned. The dictionary contains the 
        """
        # The real indices.
        idx0 = idx * 2
        idx1 = idx0 + 1

        # Load the images.
        img0 = cv2.imread( self.imageFiles[idx0] )
        img1 = cv2.imread( self.imageFiles[idx1] )

        # Load the depth files.
        dpt0 = np.load( self.depthFiles[idx0] )
        dpt1 = np.load( self.depthFiles[idx1] )

        sample = { "image0": img0, "image1": img1, "depth0": dpt0, "depth1": dpt1 }

        if ( self.transform is not None ):
            sample = self.transform( sample )

        return sample

if __name__ == "__main__":
    # Test the data loader object.
    asd = AirsimStereoDataset( "../data/airsim_oldtown_stereo_01", "image", "depth_plan" )

    # Get a sample.
    sample = asd[10]

    # Create a Downsample object.
    ds = Downsample((2, 2))
    dsSample = ds(sample)

    # Create a ToTensor object.
    tt = ToTensor()
    ttSample = tt(dsSample)

    # Test transforms.
    cm = transforms.Compose( [ds, tt] ) 
    cmSample = cm( sample )

    # Create a data loader with transform.
    asdTs = AirsimStereoDataset( "../data/airsim_oldtown_stereo_01", "image", "depth_plan", transform = cm )
    sampleTs = asdTs[10]

    # Test with data loader.
    dataLoader = DataLoader( asdTs, batch_size = 4, shuffle = False, num_workers = 4, drop_last = False )

    for idxBatch, batchedSample in enumerate( dataLoader ):
        print( "idx = {}, size = {}.".format(idxBatch, batchedSample["image0"].size()) )
