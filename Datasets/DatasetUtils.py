
from __future__ import print_function

import glob

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

def find_filenames_recursively(d, fnPattern):
    """
    Find all the filenames srecursively and starting at d. The resulting filenames
    will be stored in a list.

    d: The root directory.
    fnPattern: The file pattern like "*.json".
    return: A list contains the strings of the file names with relative paths.

    NOTE: This function heavily relys on the Python package glob. Particularly, the glob
    for Python version higher than 3.4.
    """

    # Compose the search pattern.
    s = d + "/**/" + fnPattern

    fnList = glob.glob(s, recursive = True)
    fnList.sort()

    return fnList
