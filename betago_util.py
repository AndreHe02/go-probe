import os
import shutil
import tarfile

import numpy as np

#import torch
from tqdm import tqdm
from betago.processor import *
from betago.dataloader.sampling import Sampler

from handcrafted_ft_processors import *


if __name__=='__main__':
    data_dir = 'C:/Users/andre/go-ai/data'
    processor = CapturedProcessor(data_directory=data_dir)
    for root, _, files in os.walk(data_dir):
        filenames = [fname for fname in files if fname.endswith('tar.gz')]
        for filename in filenames:
            if filename.endswith('tar.gz'):
                print(filename)
                processor.process_zip_full(root, filename, filename.split('.')[0]+'preprocessed', write_fts=False)

    # /home/yangk/data/KGS KGS-2016_05-19-1011-.tar.gz KGS-2016_05-19-1011-train
