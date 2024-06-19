#!/usr/bin/env python

"""
Corrections module for the microscopy acquisition

corrections available:
#. Dark-field correction
    * Always applicable and should be always performed
#. Flat-field, ie bright-field correction
    * Important for transmission measurements. Performs badly
        on the floating elements in the FOV
    * Harder to do for the Fluorescence, because of the excitation scatter
        leaking through filters.
    * Flat field can be used to identify dead pixels too
#. Hot pixel correction
    * Hot pixels obtained from long exposure on the blocked camera
    * Dead pixels form flat field acquisition.
    * TODO Notimplemented: Alternative you can use it for dead pixel
        identification (rename to BadPixel correction)
    * Possible to correct as mean of 4 or 8 neighbours
TODO: Intensity correction
More crucial for the shorter exposure times (less averaging of the source
intensity variations), which depends on which time-scale the light-source
drifts.

Notes:
* Exposure on the dark and bright field corrections must be
the same as the experimental exposure.
* TODO: Need to ensure correct logic of the correction and their redefinition.
    Is the logic supposed to be taken care of on this class level?
"""

import numpy as np
from utils import is_positive, img_to_int_type


def clip_img(img, messageBox=None):
    if np.amax(img) > 1 or np.amin(img) < 0:
        try:
            messageBox.setText(
                f'Image values out of range. Clipping to 0-1. Overflows: {img.min()}, {img.max()}',
            )
        except AttributeError:
            print('Image values out of range. Clipping to 0-1.',
                  f'Overflows: {img.min()}, {img.max()}')
        img = np.clip(img, 0, 1)

    return img


def convert01_to_uint16(img: np.ndarray):
    if np.amax(img) > 1 or np.amin(img) < 0:
        raise ValueError('Image is not between 0 and 1.')

    return (img * 65535).astype(np.uint16)


def subtract_images(image, corr, messageBox=None):
    if (not np.issubdtype(image.dtype, np.integer) or
            not np.issubdtype(corr.dtype, np.integer)):
        try:
            messageBox.setText(
                'Either data or corr is not np.integer type.',
            )
        except AttributeError:
            print('Either data or corr is not np.integer type.')

        data_corr = np.round((image - corr.clip(None, image))
                             ).astype(np.uint16)
    else:
        data_corr = image - corr

    return data_corr


def apply_corr_dark_bright(original_image: np.ndarray, dark=None, bright=None,
                      flagExp='Transmission', flagDark=False, flagBright=False,
                      messageBox=None):

    data_corr = np.empty(original_image.shape)

    if flagBright:
        if flagDark and flagExp == 'Transmission':
            data_corr = ((original_image - dark) / (bright - dark))
            data_corr = clip_img(data_corr, messageBox=messageBox)
            data_corr = convert01_to_uint16(data_corr)

        elif flagExp == 'Emission':
            data_corr = subtract_images(original_image, bright,
                                        messageBox=messageBox)
            data_corr = np.clip(data_corr, 0, None).astype(np.uint16)

        else:  # transmission, no dark correction
            data_corr = original_image / bright
            data_corr = clip_img(data_corr, messageBox=messageBox)
            data_corr = convert01_to_uint16(data_corr)

    elif flagDark:  # only dark correction for both Tr and Em
        data_corr = subtract_images(original_image, dark,
                                    messageBox=messageBox)

    else:
        try:
            messageBox.setText('No correction applied.')
        except AttributeError:
            print('No correction applied.')

    return data_corr.astype(np.uint16)


def apply_corr_int(img_stack: np.ndarray, mode: str = 'integral',
                   bright: np.ndarray = None, rect_dim: int = 50,
                   cast_to_int: bool = True) -> np.ndarray:
    """Intensity correction over the stack of images which are expected
    to have the same background intensity. It is preferable to
    corrected for dark, bright, and bad pixels first

    Args:
        img_stack (np.array): 3D array of images, third dimension is
            along angles
        mode (str, optional): correction mode, only available is integral.
            Defaults to 'integral'.
        use_bright (bool, optional): if bright field acquisition is a ref
            to scale images. Defaults to True.
        rect_dim (int, optional): size of rectangles in the corners in
            pixels. Defaults to 50.

    Raises:
        NotImplementedError: Checking available correction modes

    Returns:
        np.ndarray: 3D array of the same shape as the img_stack,
            but intensity corrected
    """
    # check if stack is 3D array
    if img_stack.ndim != 3:
        raise IndexError('Stack has to have three dimensions.')
    # do I want to correct in respect to the bright field
    # basic idea is four corners, integrated
    # second idea, fit a correction plane into the four corners.
    if bright is not None:
        # four corners of the bright
        ref = ((bright[:rect_dim, :rect_dim]),
               (bright[:rect_dim, -rect_dim:]),
               (bright[-rect_dim:, :rect_dim]),
               (bright[-rect_dim:, -rect_dim:]),
               )
    else:
        print('Using avg of the corners in the img stack as ref')
        # assuming the stacks 3rd dimension is the right one.
        # mean over steps in the aquisition
        ref = ((np.mean(img_stack[:, :rect_dim, :rect_dim], axis=0)),
               (np.mean(img_stack[:, :rect_dim, -rect_dim:], axis=0)),
               (np.mean(img_stack[:, -rect_dim:, :rect_dim], axis=0)),
               (np.mean(img_stack[:, -rect_dim:, -rect_dim:], axis=0)),
               )

    print('shape ref:', [k.shape for k in ref])

    # integral takes sum over pixels of interest
    # TODO: This if else structure is cumbersome
    if mode == 'integral':
        # sum of all pixels over all four squares
        # this is one number
        ref_mean = np.mean([np.mean(k) for k in ref])
    elif mode == 'integral_bottom':
        ref_mean = np.mean([np.mean(ref[2]), np.mean(ref[3])])
    else:
        raise NotImplementedError

    # correct the stack
    corr_stack = np.empty(img_stack.shape, dtype=img_stack.dtype)

    # intensity numbers for img in the stack (sum over ROIs)
    stack_int = []
    # intensity numbers for img in the stack (sum over ROIs)
    intOrig, intCorr = [], []
    for i, img in enumerate(img_stack):
        # two means are not a clean solution
        # as long as the rectangles ar the same, it is equivalent
        if mode == 'integral':
            img_int = np.mean((np.mean(img[:rect_dim, :rect_dim]),
                               np.mean(img[:rect_dim, -rect_dim:]),
                               np.mean(img[-rect_dim:, :rect_dim]),
                               np.mean(img[-rect_dim:, -rect_dim:]),
                               ))
        elif mode == 'integral_bottom':
            img_int = np.mean((
                            np.mean(img[-rect_dim:, :rect_dim]),
                            np.mean(img[-rect_dim:, -rect_dim:]),
            ))
        else:
            raise NotImplementedError
        intOrig.append(img_int)
        stack_int.append(img_int)
        corr_stack[i] = (img / img_int) * ref_mean

        # intensity after correction
        intCorr.append(np.mean((np.mean(corr_stack[i][:rect_dim, :rect_dim]),
                                np.mean(corr_stack[i][:rect_dim, -rect_dim:]),
                                np.mean(corr_stack[i][-rect_dim:, :rect_dim]),
                                np.mean(corr_stack[i][-rect_dim:, -rect_dim:]),
                                ))
                       )
        print(i, end='\r')

    # stored in order to tract the stability fluctuations.
    intOrig = np.array(intOrig)
    intCorr = np.array(intCorr)

    # test for negative values
    is_positive(corr_stack, 'Intensity')

    # cast it on correct dtype
    if cast_to_int:
        corr_stack = img_to_int_type(corr_stack, dtype=corr_stack.dtype)

    # int correction dictionary report
    intCorrReport = {'mode': mode,
                     'bright': bright,
                     'rect_dim': rect_dim,
                     'ref': ref_mean,
                     'stack_orig_int': intOrig,
                     'stack_corr_int': intCorr,
                     }

    return corr_stack, intCorrReport
