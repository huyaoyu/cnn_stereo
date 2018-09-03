
from __future__ import print_function

import cv2
import glob
import os

from torch.utils.data import Dataset, DataLoader

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

class AirsimStereoDataset(Dataset):
    def __init__(self, rootDir, imageDir, depthDir, imageExt = "png", depthExt = "npy"):
        self.rootDir  = rootDir
        self.imageDir = imageDir
        self.depthDir = depthDir
        self.imageExt = imageExt # File extension of image file.
        self.depthExt = depthExt # File extension of depth file.

        # Find all the file names.

    
