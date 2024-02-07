import numpy as np
from napari.layers import Image, Shapes, Points
from magicgui import magic_factory
from napari import Viewer
from napari.utils import notifications

from typing import Annotated
from napari.qt.threading import thread_worker
import warnings

from utils import (
    select_roi,
    bin_stack_faster,
)


# ROI
@magic_factory(call_button="Select ROI")
def select_ROIs(viewer: Viewer,
                image: Image,
                points_layer: Points,
                roi_height: Annotated[int, {'max': 3000}] = 50,
                roi_width: Annotated[int, {'max': 3000}] = 50,
                ):
    original_stack = np.asarray(image.data)
    points = np.asarray(points_layer.data)
    notifications.show_info(f'UL corner coordinates: {points[0][1:].astype(int)}')

    selected_roi = select_roi(original_stack,
                              points[0],
                              roi_height,
                              roi_width)
    viewer.add_image(selected_roi)


# Binning
@magic_factory(call_button="Bin")
def bin_stack(viewer: Viewer,
              image: Image,
              bin_factor: Annotated[int, {'min': 1, 'max': 50}] = 2,
              ):
    if bin_factor != 1:
        binned_roi = bin_stack_faster(image.data, bin_factor)
        notifications.show_info(f'Original shape: {image.data.shape}, binned shape: {binned_roi.shape}')
        viewer.add_image(binned_roi)
    else:
        notifications.show_info('Bin factor is 1, nothing to do.')


# Image corrections
@magic_factory(call_button="Image Corrections")
def corrections(viewer: Viewer,
                image: Image,
                dark: Image,
                bright: Image,
                ):
    original_stack = np.asarray(image.data)

    # first correct bright field for dark
    bright_corr = bright.data - dark.data

    ans = np.empty(original_stack.shape)
    for i, img in enumerate(original_stack):
        ans[i] = img - dark.data - bright_corr

    viewer.add_image(ans)


if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    roi_widget = select_ROIs()
    correct_widget = corrections()
    bin_widget = bin_stack()

    viewer.window.add_dock_widget(roi_widget, name='select ROI',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(bin_widget, name='Bin Stack',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(correct_widget, name='Image Corrections',
                                  area='right', add_vertical_stretch=True)
    warnings.filterwarnings('ignore')
    napari.run()
