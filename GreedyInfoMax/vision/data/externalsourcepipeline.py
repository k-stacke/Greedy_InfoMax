import types
import numpy as np
import collections
import pandas as pd

from random import shuffle

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class ExternalSourcePipeline(Pipeline):
    '''
    Used by dali.pipeline to input the data and feed it to device with augmentations
    input: data_iterator - dali specific iterator that batches the data
    '''
    def __init__(self, data_iterator, batch_size, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.data_iterator = data_iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

        # Define augmentations
        self.norm = ops.CropMirrorNormalize(device="gpu", output_layout='CWH',
                                            mean=[125.31, 122.95, 113.87],
                                            std=[63.0, 62.09, 66.70])
        self.coin = ops.CoinFlip(probability=0.5)
        self.coin2 = ops.CoinFlip(probability=0.5)
        self.flip = ops.Flip(device="gpu")
        self.hsv = ops.Hsv(device = "gpu")
        self.hue_random = ops.Uniform(range=[-40, 40]) #pretty large variations

        self.pos_rng_x = ops.Uniform(range = (0.0, 1.0))
        self.pos_rng_y = ops.Uniform(range = (0.0, 1.0))
        self.decode = ops.ImageDecoderCrop(device = 'mixed',  
                                           crop = (64, 64))

        # resizing is *must* because loaded images maybe of different sizes
        # and to create GPU tensors we need image arrays to be of same size
        #self.res = ops.Resize(device="gpu", resize_x=64, resize_y=64, interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()

        # The augmentations are applied
        pos_x = self.pos_rng_x()
        pos_y = self.pos_rng_y()
        images = self.decode(self.jpegs, crop_pos_x=pos_x, crop_pos_y=pos_y)
        images = self.flip(images, horizontal = self.coin(), vertical = self.coin2())
        hue_random = self.hue_random()
        images = self.hsv(images, hue=hue_random)
         
        output = self.norm(images)
        return (output, self.labels)

    def iter_setup(self):
        # the external data iterator is consumed here and fed as input to Pipeline
        images, labels = self.data_iterator.__next__()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)
