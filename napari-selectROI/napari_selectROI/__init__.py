#!/usr/bin/env python


''' Widget to select ROI'''


import numpy as np
from napari.layers import Image, Shapes, Points
from magicgui import magicgui
from napari import Viewer
from typing import Annotated

from .widget_funcs import (
    select_roi,
    bin_stack_faster
)


def select_ROI(viewer: Viewer,
               image: Image,
               points_layer: Points,
               roi_height: Annotated[int, {'max': 3000}] = 50,
               roi_width: Annotated[int, {'max': 3000}] = 50,
               bin_factor: int = 2,
               ):
    original_stack = np.asarray(image.data, dtype=np.int16)
    points = np.asarray(points_layer.data)
    print(points[0])

    selected_roi = select_roi(original_stack,
                              points[0],
                              roi_height,
                              roi_width)
    # binning option
    if bin_factor != 1:
        binned_roi = bin_stack_faster(selected_roi, bin_factor)
        print(selected_roi.shape, binned_roi.shape)
        viewer.add_image(binned_roi)
    else:
        viewer.add_image(selected_roi)
