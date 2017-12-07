#!/usr/bin/env python

# Copyright 2017 (Author: Chun Chieh Chang, Ashish Arora)

""" This module will be used for line image preprocessing. It
    performs image resizing while maintaining the aspect ratio,
    image padding on left and right side of the image and scales
    image pixel values between 0 and 1. It will then write the
    raw pixel features in kaldi format.
  Args:
   dir: utterance id and image path directory
   --out-ark: location of output features.
   --scale-size: size to scale the height of all images.
   --padding: width of white pixels on lift and right side of image

  Eg. local/make_feature.py data/train --scale-size 40
"""

import argparse
import os
import sys
import scipy.io as sio
import numpy as np
from scipy import misc

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

parser = argparse.ArgumentParser(description="""Generates and saves the feature vectors""")
parser.add_argument('dir', type=str, help='directory of images.scp')
parser.add_argument('--out-ark', type=str, default='-', help='where to write the output feature file')
parser.add_argument('--scale-size', type=int, default=40, help='size to scale the height of all images')
parser.add_argument('--padding', type=int, default=5, help='width of white pixels')
args = parser.parse_args()


def write_kaldi_matrix(file_handle, matrix, key):
    file_handle.write(key + " [ ")
    num_rows = len(matrix)
    if num_rows == 0:
        raise Exception("Matrix is empty")
    num_cols = len(matrix[0])

    for row_index in range(len(matrix)):
        if num_cols != len(matrix[row_index]):
            raise Exception("All the rows of a matrix are expected to "
                            "have the same length")
        file_handle.write(" ".join(map(lambda x: str(x), matrix[row_index])))
        if row_index != num_rows - 1:
            file_handle.write("\n")
    file_handle.write(" ]\n")

def get_scaled_image(im):
    scale_size = args.scale_size
    sx = im.shape[1]
    sy = im.shape[0]
    scale = (1.0 * scale_size) / sy
    nx = int(scale_size)
    ny = int(scale * sx)
    im = misc.imresize(im, (nx, ny))
    padding_x = max(5,int((args.padding/100)*im.shape[1]))
    padding_y = im.shape[0]
    im_pad = np.concatenate((255 * np.ones((padding_y,padding_x), dtype=int), im), axis=1)
    im_pad1 = np.concatenate((im_pad,255 * np.ones((padding_y, padding_x), dtype=int)), axis=1)
    return im_pad1

### main ###
data_list_path = os.path.join(args.dir,'images.scp')

if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark,'wb')

with open(data_list_path) as f:
    for line in f:
        line = line.strip()
        line_vect = line.split(' ')
        image_id = line_vect[0]
        image_path = line_vect[1]
        im = misc.imread(image_path)
        im_scale = get_scaled_image(im)
        
        data = np.transpose(im_scale, (1, 0))
        data = np.divide(data, 255.0)
        write_kaldi_matrix(out_fh, data, image_id)

