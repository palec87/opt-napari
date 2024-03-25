#!/usr/bin/env python

import numpy as np
import warnings
from napari.utils import progress, notifications


"""
Utility functions for the widget and for the Correct class
TODO: perform timing on real data and consider threading.
"""

def select_roi(stack: np.ndarray,
               ul_corner,
               height: int,
               width: int) -> tuple[np.ndarray, tuple]:
    """
    Select ROI relative to Upper Left (UL) corner point. If points
    layer contains more than 1 point, the last point is considered.

    Args:
        stack (np.ndarray): Stack of images, firs dim is stacking dim
        ul_corner (napari point): napari point layer
        height (int): ROI's height
        width (int): ROI's

    Raises:
        IndexError: If height or width exceeds the limits of the images.
            In that case largest possible ROI is taken and warning is shown.

    Returns:
        np.ndarray: ROIs from the stack
    """

    _, ix, iy = ul_corner.astype(int)

    # takes care of ROI beyond the image size
    try:
        roi = stack[:,
                    ix: ix + height,
                    iy: iy + width]
        roi_pars = (ix, ix + height, iy, iy + width)
    except IndexError:
        notifications.show_warning(
            'Too large ROI for the image, selecting maximum possible size.',
            )
        iix = min(ix + height, stack.shape[1])
        iiy = min(iy + width, stack.shape[2])
        roi = stack[:,
                    ix: iix,
                    iy: iiy]
        roi_pars = (ix, iix, iy, iiy)
    return roi, roi_pars

# def select_roi(stack: np.ndarray,
#                ul_corner, height: int,
#                width: int) -> tuple[np.ndarray, tuple]:
#     """
#     Select ROI relative to Upper Left (UL) corner point. If points
#     layer contains more than 1 point, the last point is considered.

#     Args:
#         stack (np.ndarray): Stack of images, firs dim is stacking dim
#         ul_corner (napari point): napari point layer
#         height (int): ROI's height
#         width (int): ROI's

#     Raises:
#         IndexError: If height or width exceeds the limits of the images.
#             In that case largest possible ROI is taken and warning is shown.

#     Returns:
#         np.ndarray: ROIs from the stack
#     """
#     _, ix, iy = ul_corner.astype(int)

#     # takes care of ROI beyond the image size
#     try:
#         roi = stack[:,
#                     ix: ix + height,
#                     iy: iy + width]
#         roi_pars = (ix, ix + height, iy, iy + width)
#     except IndexError:
#         notifications.show_warning(
#             'Too large ROI for the image, selecting maximum possible size.',
#             )
#         iix = min(ix + height, stack.shape[1])
#         iiy = min(iy + width, stack.shape[2])
#         roi = stack[:,
#                     ix: iix,
#                     iy: iiy]
#         roi_pars = (ix, iix, iy, iiy)
#     return roi, roi_pars


# TODO: should there be sum method allowed?
def bin_3d(stack: np.ndarray, bin_factor: int) -> np.ndarray:
    """
    Bin stack of images applying mean on the binned pixels.
    First dim is the stacking one. Binning is along axis 1, 2.
    Result is casted on integer.

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

    # array preallocation
    height_dim = stack.shape[1] // bin_factor
    width_dim = stack.shape[2] // bin_factor
    ans = np.empty((stack.shape[0], height_dim, width_dim),
                   dtype=int,
                   )
    # TODO: this throws error if reshape not exact, needs a fix.
    for i in progress(range(stack.shape[0])):
        ans[i] = stack[i].reshape(height_dim, bin_factor,
                                  width_dim, bin_factor).mean(3).mean(1)
    return ans


######################
# Funcs used in Correct class
######################
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
    return img/np.amax(img) if ret_type == 'float'else (img/np.amax(img)).astype(ret_type)


def img_to_int_type(img: np.array, dtype: np.dtype = np.int_) -> np.array:
    """
    After corrections, resulting array can be dtype float. Two steps are
    taken here. First convert to a chosed dtype and then clip values as if it
    was unsigned int, which the images are.

    Args:
        img (np.array): img to convert
        dtype (np.dtype): either np.int8 or np.int16 currently,
            Defaults to np.int_

    Returns:
        np.array: array of type int
    """
    # TODO: take care of 12 bit images, how to identify them in order
    # to normalize on 2**12-1 but witll on 16bit. Memory saving in practice
    if dtype == np.int8:
        ans = np.clip(img, 0, 255).astype(dtype)
    elif dtype == np.int16:
        # 4095 would be better for our 12bit camera
        ans = np.clip(img, 0, 2**16 - 1).astype(dtype)
    else:
        ans = np.clip(img, 0, np.amax(img)).astype(np.int_)

    return ans


def is_positive(img, corr_type='Unknown'):
    if np.any(img < 0):
        warnings.warn(
            f'{corr_type} correction: Some pixel < 0, casting them to 0.',
            )
