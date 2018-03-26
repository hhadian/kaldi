#!/usr/bin/env python
import random
import argparse
import os
import sys
import scipy.io as sio
import numpy as np
from scipy import misc
from scipy.ndimage.interpolation import affine_transform
import math
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

parser = argparse.ArgumentParser(description="""Generates and saves
                                 the feature vectors""")
parser.add_argument('dir', type=str,
                    help='directory of images.scp and is also output directory')
parser.add_argument('--job', type=str, default='',
                    help='JOB number of images.JOB.scp')
parser.add_argument('--out-ark', type=str, default='-',
                    help='where to write the output feature file')
parser.add_argument('--feat-dim', type=int, default=40,
                    help='size to scale the height of all images')
parser.add_argument('--padding', type=int, default=5,
                    help='size to scale the height of all images')
parser.add_argument('--augment', type=str, default='false',
                    help='whether or not to do image augmentation on training set')
parser.add_argument('--vertical-shift', type=int, default=10,
                    help='total number of padding pixel per column')
parser.add_argument('--horizontal-shear', type=int, default=45,
                    help='maximum horizontal shearing degree')
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
    scale_size = args.feat_dim
    sx = im.shape[1]  # width
    sy = im.shape[0]  # height
    scale = (1.0 * scale_size) / sy
    nx = int(scale_size)
    ny = int(scale * sx)
    im = misc.imresize(im, (nx, ny))
    padding_x = max(5, int((args.padding / 100) * im.shape[1]))
    padding_y = im.shape[0]
    im_pad = np.concatenate(
        (255 * np.ones((padding_y, padding_x), dtype=int), im), axis=1)
    im_pad1 = np.concatenate(
        (im_pad, 255 * np.ones((padding_y, padding_x), dtype=int)), axis=1)
    return im_pad1


def contrast_normalization(im, low_pct, high_pct):
    element_number = im.size
    rows = im.shape[0]
    cols = im.shape[1]
    im_contrast = np.zeros(shape=im.shape)
    low_index = int(low_pct * element_number)
    high_index = int(high_pct * element_number)
    sorted_im = np.sort(im, axis=None)
    low_thred = sorted_im[low_index]
    high_thred = sorted_im[high_index]
    for i in range(rows):
        for j in range(cols):
            if im[i, j] > high_thred:
                im_contrast[i, j] = 255  # lightest to white
            elif im[i, j] < low_thred:
                im_contrast[i, j] = 0  # darkest to black
            else:
                # linear normalization
                im_contrast[i, j] = (im[i, j] - low_thred) * \
                    255 / (high_thred - low_thred)
    return im_contrast


def geometric_moment(frame, p, q):
    m = 0
    for i in range(frame.shape[1]):
        for j in range(frame.shape[0]):
            m += (i ** p) * (j ** q) * frame[i][i]
    return m


def central_moment(frame, p, q):
    u = 0
    x_bar = geometric_moment(frame, 1, 0) / \
        geometric_moment(frame, 0, 0)  # m10/m00
    y_bar = geometric_moment(frame, 0, 1) / \
        geometric_moment(frame, 0, 0)  # m01/m00
    for i in range(frame.shape[1]):
        for j in range(frame.shape[0]):
            u += ((i - x_bar)**p) * ((j - y_bar)**q) * frame[i][j]
    return u


def height_normalization(frame, w, h):
    frame_normalized = np.zeros(shape=(h, w))
    alpha = 4
    x_bar = geometric_moment(frame, 1, 0) / \
        geometric_moment(frame, 0, 0)  # m10/m00
    y_bar = geometric_moment(frame, 0, 1) / \
        geometric_moment(frame, 0, 0)  # m01/m00
    sigma_x = (alpha * ((central_moment(frame, 2, 0) /
                         geometric_moment(frame, 0, 0)) ** .5))  # alpha * sqrt(u20/m00)
    sigma_y = (alpha * ((central_moment(frame, 0, 2) /
                         geometric_moment(frame, 0, 0)) ** .5))  # alpha * sqrt(u02/m00)
    for x in range(w):
        for y in range(h):
            i = int((x / w - 0.5) * sigma_x + x_bar)
            j = int((y / h - 0.5) * sigma_y + y_bar)
            frame_normalized[x][y] = frame[i][j]
    return frame_normalized


def find_slant_project(im):
    rows = im.shape[0]
    cols = im.shape[1]
    std_max = 0
    alpha_max = 0
    col_disp = np.zeros(90, int)
    proj = np.zeros(shape=(90, cols + 2 * rows), dtype=int)
    for r in range(rows):
        for alpha in range(-45, 45, 1):
            col_disp[alpha] = int(r * math.tan(alpha / 180.0 * math.pi))
        for c in range(cols):
            if im[r, c] < 100:
                for alpha in range(-45, 45, 1):
                    proj[alpha + 45, c + col_disp[alpha] + rows] += 1
    for alpha in range(-45, 45, 1):
        proj_histogram, bin_array = np.histogram(proj[alpha + 45, :], bins=10)
        proj_std = np.std(proj_histogram)
        if proj_std > std_max:
            std_max = proj_std
            alpha_max = alpha
    proj_std = np.std(proj, axis=1)
    return -alpha_max


def horizontal_shear(im, degree):
    rad = degree / 180.0 * math.pi
    padding_x = int(abs(np.tan(rad)) * im.shape[0])
    padding_y = im.shape[0]
    if rad > 0:
        im_pad = np.concatenate(
            (255 * np.ones((padding_y, padding_x), dtype=int), im), axis=1)
    elif rad < 0:
        im_pad = np.concatenate(
            (im, 255 * np.ones((padding_y, padding_x), dtype=int)), axis=1)
    else:
        im_pad = im
    shear_matrix = np.array([[1, 0],
                             [np.tan(rad), 1]])
    # sheared_im = affine_transform(image, shear_matrix, output_shape=(
    # im.shape[0], im.shape[1] + abs(int(im.shape[0] * np.tan(shear)))), cval=128.0)
    sheared_im = affine_transform(im_pad, shear_matrix, cval=255.0)
    return sheared_im


def vertical_shift(im, mode='mid'):
    total = args.vertical_shift
    if mode == 'mid':
        top = total / 2
        bottom = total - top
    elif mode == 'top':  # more padding on top
        top = random.randint(total / 2, total)
        bottom = total - top
    elif mode == 'bottom':  # more padding on bottom
        top = random.randint(0, total / 2)
        bottom = total - top
    width = im.shape[1]
    im_pad = np.concatenate(
        (255 * np.ones((top, width), dtype=int) -
         np.random.normal(2, 1, (top, width)).astype(int), im), axis=0)
    im_pad = np.concatenate(
        (im_pad, 255 * np.ones((bottom, width), dtype=int) -
         np.random.normal(2, 1, (bottom, width)).astype(int)), axis=0)
    return im_pad


def image_augment(im, out_fh, image_id):
    # shift_setting = ['mid', 'top', 'bottom']
    slant_degree = find_slant_project(im)
    shear_degrees = [0, random.randint(0, args.horizontal_shear),
                     random.randint(-args.horizontal_shear, 0)]
    im_deslanted = horizontal_shear(im, slant_degree)
    image_shear_id = []
    for i in range(3):
        image_shear_id.append(image_id + '_shear' + str(i + 1))
        im_shear = horizontal_shear(im_deslanted, shear_degrees[i])
        data = np.transpose(im_shear, (1, 0))
        data = np.divide(data, 255.0)
        write_kaldi_matrix(out_fh, data, image_shear_id[i])

        # image_shift_id.append(image_id + '_shift' + str(i + 1))
        # im_shift = vertical_shift(im, shift_setting[i])
        # data = np.transpose(im_shift, (1, 0))
        # data = np.divide(data, 255.0)
        # write_kaldi_matrix(out_fh, data, image_shift_id[i])


# main #

random.seed(1)

scp_name = 'images.scp'  # parallel
data_list_path = os.path.join(args.dir, scp_name)
# output dir of feature matrix
if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark, 'wb')


if (args.augment == 'true') and ('train' in args.dir):
    # only do image augmentation for training data
    with open(data_list_path) as f:
        for line in f:
            line = line.strip()
            line_vect = line.split(' ')
            image_id = line_vect[0]
            image_path = line_vect[1]
            im = misc.imread(image_path)
            im_contrast = contrast_normalization(im, 0.05, 0.2)
            im_scaled = get_scaled_image(im)
            image_augment(im_scaled, out_fh, image_id)

else:  # settings for without augmentation or test data
    with open(data_list_path) as f:
        for line in f:
            line = line.strip()
            line_vect = line.split(' ')
            image_id = line_vect[0]
            image_path = line_vect[1]
            im = misc.imread(image_path)
            im_scaled = get_scaled_image(im)
            im_contrast = contrast_normalization(im_scaled, 0.05, 0.2)
            slant_degree = find_slant_project(im_contrast)
            im_sheared = horizontal_shear(im_contrast, slant_degree)
            im_padded = vertical_shift(im_sheared)
            data = np.transpose(im_padded, (1, 0))
            data = np.divide(data, 255.0)
            write_kaldi_matrix(out_fh, data, image_id)
