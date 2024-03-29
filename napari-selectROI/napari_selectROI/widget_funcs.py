#!/usr/bin/env python

import numpy as np
from tqdm import tqdm


def select_roi(stack, ur_corner, height, width):
    _, ix, iy = ur_corner.astype(int)
    roi = stack[:,
                ix: ix + height,
                iy: iy + width]
    return roi


def bin_stack_faster(stack, bin_factor):
    if len(stack.shape) != 3:
        raise IndexError('Stack has to have three dimensions.')

    height_dim = stack.shape[1] // bin_factor
    width_dim = stack.shape[2] // bin_factor
    ans = np.empty((stack.shape[0],
                    height_dim,
                    width_dim),
                   dtype=np.int16)
    for i in tqdm(range(stack.shape[0])):
        ans[i] = stack[i].reshape(height_dim, bin_factor,
                                  width_dim, bin_factor).mean(3).mean(1)
    return ans
