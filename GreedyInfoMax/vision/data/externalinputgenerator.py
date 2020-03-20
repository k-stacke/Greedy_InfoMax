import types
import numpy as np
import collections
import pandas as pd

import random

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class ExternalInputIterator(object):
    '''
    Used by DALI to read files from disk and handle batching
    '''
    def __init__(self, batch_size, data_frame, image_dir, shuffle=True):
       self.img_dir = image_dir
       self.batch_size = batch_size
       self.data_frame = data_frame


       #self.files = [{'label': row.label_int, 'filename': row.filename} for _, row in data_frame.iterrows()]
       self.files = [{'label': row.center, 'filename': row.filename} for _, row in data_frame.iterrows()]
       if shuffle:
           random.shuffle(self.files)


    def __iter__(self):
       self.i = 0
       self.n = len(self.files)
       return self

    def __len__(self):
       return len(self.files)


    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            data = self.files[self.i]
            jpeg_filename = data['filename']
            label = data['label']
            
            f = open(f'{self.img_dir}/{jpeg_filename}', 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array(label, dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return batch, labels
