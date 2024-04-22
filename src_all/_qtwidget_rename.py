import numpy as np
# from napari import Viewer
# from typing import Annotated
# import os
#import processors
# from qtpy.QtCore import Qt
# import numpy as np
# from napari.qt.threading import thread_worker
# import warnings
# from time import time
# import scipy.ndimage as ndi
# import cv2 
# import tqdm
from corrections import Correct
# from dataclasses import dataclass
from napari.viewer import current_viewer
from typing import List
from functools import partial

from napari.layers import Layer, Image, Points
from magicgui import magic_factory
from magicgui.widgets import create_widget
from napari.utils import progress, notifications
import warnings
from widget_settings import Settings, Combo_box
import napari
from qtpy.QtWidgets import (
    QVBoxLayout, QSplitter, QHBoxLayout,
    QWidget, QPushButton, QLineEdit, QSpinBox,
    QDoubleSpinBox, QFormLayout, QComboBox, QLabel,
)
from qtpy.QtCore import QEvent, QObject
from enum  import Enum
from qtpy.QtCore import Qt

from utils import (
    select_roi,
    bin_3d,
)

def layer_container_and_selection(viewer=None, layer_type = Image, container_name = 'Layer'):
    """
    Create a container and a dropdown widget to select the layer.

    Returns
    -------
    A tuple containing a QWidget for displaying the layer selection container,
    and a QWidget containing the selection options for the layer.
    """
    layer_selection_container = QWidget()
    layer_selection_container.setLayout(QHBoxLayout())
    layer_selection_container.layout().addWidget(QLabel(container_name))
    layer_select = create_widget(annotation= layer_type, label="layer")
    layer_selection_container.layout().addWidget(layer_select.native)

    if viewer is not None and viewer.layers.selection.active is not None:
        layer_select.value = viewer.layers.selection.active

    return layer_selection_container, layer_select

class PreprocessingnWidget(QWidget):
    name = 'Preprocessor'

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.setup_ui()
    
    def setup_ui(self):
        
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        mode_layout = QVBoxLayout()
        layout.addLayout(mode_layout)
        self.choose_mode(mode_layout)

        (image_selection_container, 
         self.image_layer_select) = layer_container_and_selection(viewer=self.viewer, 
                                                                  layer_type = Image, 
                                                                  container_name = 'Image to analyze')
        
        (hot_image_selection_container, 
         self.hot_layer_select) = layer_container_and_selection(viewer=self.viewer, 
                                                                layer_type = Image, 
                                                                container_name = 'Hot correction image')
        (dark_image_selection_container, 
         self.dark_layer_select) = layer_container_and_selection(viewer=self.viewer, 
                                                                 layer_type = Image, 
                                                                 container_name = 'Dark correction image')
        (bright_image_selection_container, 
         self.bright_layer_select) = layer_container_and_selection(viewer=self.viewer, 
                                                                   layer_type = Image, 
                                                                   container_name = 'Bright correction image')
        (points_selection_container, 
         self.points_layer_select) = layer_container_and_selection(viewer=self.viewer, 
                                                                   layer_type = Points, 
                                                                   container_name = 'Points layer for ROI selection')
        
        image_layout = QVBoxLayout()
        layout.addLayout(image_layout)
        self.layout().addWidget(image_selection_container)
        self.layout().addWidget(hot_image_selection_container)
        self.layout().addWidget(dark_image_selection_container)
        self.layout().addWidget(bright_image_selection_container)
        self.layout().addWidget(points_selection_container)
        
        for container in [image_selection_container, hot_image_selection_container,
                      dark_image_selection_container, bright_image_selection_container,
                      points_selection_container]:
            container.layout().setContentsMargins(0, 0, 0, 10)

        settings_layout = QVBoxLayout()
        layout.addLayout(settings_layout)
        self.createSettings(settings_layout)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.image_layer_select.reset_choices(event)
        self.hot_layer_select.reset_choices(event)
        self.dark_layer_select.reset_choices(event)
        self.bright_layer_select.reset_choices(event)
        self.points_layer_select.reset_choices(event)
        
    def choose_mode(self,slayout):
        
        self.inplace = Settings('Inplace operations', dtype=bool, initial=False,
                                      layout=slayout, 
                                      write_function = self.set_preprocessing)
        
        self.track = Settings('Track', dtype=bool, initial=False, #TODO track box must appear only when inplace box=True
                                      layout=slayout,
                                      write_function = self.set_preprocessing)
        
    def createSettings(self, slayout):
        
        layout = QVBoxLayout()
        
        
        self.std_cutoff = Settings('Std cutoff',
                                  dtype=int, 
                                  initial=5,
                                  layout=slayout, 
                                  write_function = self.set_preprocessing)
        
        self.roi_height = Settings('ROI height',
                                  dtype=int, 
                                  initial=200,
                                  vmin = 1,
                                  vmax = 5000,
                                  layout=slayout, 
                                  write_function = self.set_preprocessing)
        
        self.roi_width = Settings('ROI width',
                                  dtype=int, 
                                  initial=200,
                                  vmin = 1,
                                  vmax = 5000,
                                  layout=slayout, 
                                  write_function = self.set_preprocessing)

        self.bin_factor = Settings('Bin factor',
                                  dtype=int, 
                                  initial=2,
                                  vmin = 1,
                                  vmax = 500,
                                  layout=slayout, 
                                  write_function = self.set_preprocessing)    
        
        # buttons
        buttons_dict = {'Select ROI': self.select_ROIs,
                        'Binning': self.bin_stack,
                        'Hot pixel correction': self.correct_hot_pixels,
                        'Dark-field correction': self.dark_correction,
                        'Bright-field correction': self.bright_correction
                        }
        for button_name, call_function in buttons_dict.items():
            button = QPushButton(button_name)
            button.clicked.connect(call_function)
            slayout.addWidget(button)
        self.messageBox = QLineEdit()
        layout.addWidget(self.messageBox, stretch=True)
        self.messageBox.setText('Messages') #not working, I think
             
    def show_image(self, image_values, _inplace_value, fullname, contrast):
        
        # if 'scale' in kwargs.keys():    #GT: do we need to scale?
        #     scale = kwargs['scale']
        # else:
        #     scale = [1.]*image_values.ndim
        
        if _inplace_value == True:
            fullname = self.image_layer_select.value.name
            self.viewer.layers[fullname].data = image_values
            # self.viewer.layers[fullname].scale = scale
        else:  
            self.viewer.add_image(image_values,
                                  name = fullname,
                                  # scale = scale,
                                  contrast_limits = contrast,
                                  interpolation2d = 'linear')
    
    def set_preprocessing(self, *args):
        '''
        Sets preprocessing parameters
        '''

        if hasattr(self, 'h'):
            
            self.h.inplace_val = self.inplace.val
            self.h.track_val = self.track.val
            self.h.cutoff_val = self.std_cutoff.val
            self.h.height_val = self.roi_height.val
            self.h.width_val = self.roi_width.val
            self.h.binning_val = self.bin_factor.val
            
            # self.h.set_reconstruction_process()
            
    #select ROIs    
    def select_ROIs(self):
 
        original_image = self.image_layer_select.value
        data = original_image.data
        contrast_limits = original_image.contrast_limits
        self.imageRaw_name = original_image.name
        points = self.points_layer_select.value.data
        _inplace_value = self.inplace.val
        fullname = 'ROI_' + self.imageRaw_name
        notifications.show_info(f'UL corner coordinates: {points[0][1:].astype(int)}')
        selected_roi, roi_pars = select_roi(data, points[-1], self.roi_height.val, self.roi_width.val)
        self.show_image(selected_roi, _inplace_value, fullname, contrast_limits)

    #binning
    def bin_stack(self):
        if self.bin_factor != 1:
            original_image = self.image_layer_select.value
            data = original_image.data
            contrast_limits = original_image.contrast_limits
            self.imageRaw_name = original_image.name
            _inplace_value = self.inplace.val
            fullname = 'binned_' + self.imageRaw_name 
            binned_roi = bin_3d(data, self.bin_factor.val)
            notifications.show_info(f'Original shape: {data.shape}, binned shape: {binned_roi.shape}')
            self.show_image(binned_roi, _inplace_value, fullname, contrast_limits)
        else:
            notifications.show_info('Bin factor is 1, nothing to do.')
    
    #hot pixels correction (NOT WORKING YET)
    def correct_hot_pixels(self):
 
        original_image = self.image_layer_select.value
        data = original_image.data
        contrast_limits = original_image.contrast_limits
        self.imageRaw_name = original_image.name
        hot_pixels_image = self.hot_layer_select.value.data
        _inplace_value = self.inplace.val
        _std_cutoff = self.std_cutoff.val
        fullname = 'Hot correction' + self.imageRaw_name
        
        # init correction class
        corr = Correct(hot=hot_pixels_image, std_mult=_std_cutoff, dark=None, bright=None)
        # preallocate corrected array
        data_corr = np.zeros(data.shape,
                              dtype=data.dtype)
        print(f'number of hot pixels: {len(corr.hot_pxs)}')

        # Bad pixels
        print('to be implemented')
    
        # # TODO: intensity correction not implemented!!!
        # # now updating the image.
        # if history.inplace:
        #     new_data = {'operation': 'correct',
        #                 'data': data_corr,
        #                 }
        #     image.data = history.update_history(image, new_data)
        # else:
        #     viewer.add_image(data_corr)
        # print('debug corrections', history.roi_def)
        
        # self.show_image(selected_roi, _inplace_value, fullname)
        self.show_image(data_corr, _inplace_value, fullname, contrast_limits)
        
    #dark-field correction
    def dark_correction(self):
 
        original_image = self.image_layer_select.value
        data = original_image.data
        contrast_limits = original_image.contrast_limits
        self.imageRaw_name = original_image.name
        dark_image = self.dark_layer_select.value
        _inplace_value = self.inplace.val
        fullname = 'dark_correction _' + self.imageRaw_name
        
        # init correction class
        corr = Correct(hot=None, std_mult=None, dark=dark_image, bright=None)
        # preallocate corrected array
        data_corr = np.zeros(data.shape,
                             dtype=data.dtype)

        for i, img in progress(enumerate(data)):
            data_corr[i] = corr.correct_dark(img)
            print(f'max: {np.amax(data_corr)}, min: {np.amin(data_corr)}')
        notifications.show_info('Dark correction done.')
        
        self.show_image(data_corr, _inplace_value, fullname, contrast_limits)
        
        self.data_corr = data_corr
        
    #bright-field correction
    def bright_correction(self):
        
        original_image = self.image_layer_select.value
        contrast_limits = original_image.contrast_limits
        self.imageRaw_name = original_image.name
        bright_image = self.bright_layer_select.value
        dark_image = self.dark_layer_select.value
        data_corr= self.data_corr
        _inplace_value = self.inplace.val
        fullname = 'bright_correction_' + self.imageRaw_name
        
        # init correction class
        corr = Correct(hot=None, std_mult=None, dark=dark_image, bright=bright_image)

        for i, img in progress(enumerate(data_corr)):
            data_corr[i] = corr.correct_bright(img)
        notifications.show_info('Bright correction done.')
        
        self.show_image(data_corr, _inplace_value, fullname, contrast_limits)

if __name__ == '__main__':
    import napari
    
    viewer = napari.Viewer()
    mywidget = PreprocessingnWidget(viewer)
    viewer.window.add_dock_widget(mywidget, name = 'Preprocessing Widget')
    # warnings.filterwarnings('ignore')
    napari.run()