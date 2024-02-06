#!/usr/bin/env python


''' Widget for Dark and Bright field correction'''


import numpy as np
from napari.layers import Image
from napari import Viewer
from tqdm import tqdm


def correct_dark_bright(
                viewer: Viewer,
                image: Image,
                dark: Image,
                bright: Image,
                ):
    original_stack = np.asarray(image.data, dtype=np.int16)

    # first correct bright field for dark
    bright_corr = bright.data - dark.data

    ans = np.empty(original_stack.shape, dtype=np.int16)
    for i, img in tqdm(enumerate(original_stack)):
        ans[i] = img - dark.data - bright_corr

    print(ans.shape)
    viewer.add_image(ans)
