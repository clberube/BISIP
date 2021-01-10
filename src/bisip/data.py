# @Author: charles
# @Date:   2020-03-19T08:52:03-04:00
# @Last modified by:   charles
# @Last modified time: 2020-03-19T08:52:49-04:00


import os
import glob

import bisip


class DataFiles(dict):
    def __init__(self, *args, **kwargs):
        super(DataFiles, self).__init__(*args, **kwargs)
        dir = os.path.join(os.path.dirname(bisip.__file__), 'data/*.dat')
        files = sorted(glob.glob(dir))
        keys = [os.path.splitext(os.path.basename(x))[0] for x in files]
        self.update({k: v for k, v in zip(keys, files)})
