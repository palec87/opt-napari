#!/usr/bin/env python

import numpy as np
import warnings
from napari.utils import progress


def select_roi(stack: np.ndarray, ul_corner, height: int, width: int):
    """
    Select ROI relative to UL corner point.

    Args:
        stack (np.ndarray): Stack of images, firs dim is stacking dim
        ul_corner (napari point): napari point layer
        height (int): ROI's height
        width (int): ROI's

    Returns:
        np.ndarray: ROIs from the stack
    """
    _, ix, iy = ul_corner.astype(int)
    roi = stack[:,
                ix: ix + height,
                iy: iy + width]
    return roi


def bin_stack_faster(stack, bin_factor):
    """
    Bin stack of images. First dim is the stacking one.
    Binning is along axis 1, 2.

    Args:
        stack (np.ndarray): stack of images
        bin_factor (int): how many pixels are binned along x and y.
            Binning is the same along x and y.

    Raises:
        IndexError: Stack is not 3 dimensional.

    Returns:
        np.ndarray: Binned stack of images.
    """
    if len(stack.shape) != 3:
        raise IndexError('Stack has to have three dimensions.')

    height_dim = stack.shape[1] // bin_factor
    width_dim = stack.shape[2] // bin_factor
    ans = np.empty((stack.shape[0],
                    height_dim,
                    width_dim),
                   dtype=int)
    for i in progress(range(stack.shape[0])):
        ans[i] = stack[i].reshape(height_dim, bin_factor,
                                  width_dim, bin_factor).sum(3).sum(1)
    return ans


def norm_img(img: np.array, ret_type='float') -> np.array:
    """
    Normalize np.array image to 1.

    Args:
        img (np.array): img to normalize
        ret_type (str, optional): result can be casted to any valid dtype.
            Defaults to 'float'.

    Returns:
        np.array: normalized array to 1
    """
    return img/np.amax(img) if ret_type == 'float' else (img/np.amax(img)).astype(ret_type)


def img_to_int_type(img: np.array, dtype: np.dtype = np.int_) -> np.array:
    """
    After corrections, resulting array can be dtype float. Two steps are
    taken here. First convert to a chosed dtype and then clip values as if it
    was unsigned int, which the images are.shape

    Args:
        img (np.array): img to convert
        dtype (np.dtype): either np.int8 or np.int16 currently,
            Defaults to np.int_

    Returns:
        np.array: array as int
    """
    # ans = img.astype(dtype)
    if dtype == np.int8:
        ans = np.clip(img, 0, 255).astype(dtype)
    elif dtype == np.int16:
        # 4095 would be better for our 12bit camera
        ans = np.clip(img, 0, 2**16 - 1).astype(dtype)
    else:
        ans = np.clip(img, 0, np.amax(img)).astype(np.int_)

    return ans


def is_positive(img):
    if np.any(img < 0):
        warnings.warn('Dark-field correction: Some pixel are negative, casting them to 0.')
