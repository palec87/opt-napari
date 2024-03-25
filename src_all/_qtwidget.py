# import numpy as np
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
# from corrections import Correct
# from dataclasses import dataclass
from napari.viewer import current_viewer
from typing import List

from napari.layers import Layer, Image, Points
from magicgui import magic_factory
from napari.utils import notifications
import warnings 
from widget_settings import Settings, Combo_box
import napari
from qtpy.QtWidgets import QVBoxLayout, QSplitter, QHBoxLayout, QWidget, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QLabel
from enum  import Enum

from utils import (
    select_roi,
    bin_3d,
)

class PreprocessingnWidget(QWidget):
    name = 'Preprocessor'
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.setup_ui()
        # self.start_preprocessing()
        # self.viewer.dims.events.current_step.connect(self.select_index)
    
    def setup_ui(self):
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        def add_section(_layout, _title):
            from qtpy.QtCore import Qt
            splitter = QSplitter(Qt.Vertical)
            _layout.addWidget(splitter)
            # _layout.addWidget(QLabel(_title))        
        
        image_layout = QVBoxLayout()
        add_section(image_layout,'Image selection')
        layout.addLayout(image_layout)
        
        self.multiple_layers_widget = multiple_layers()
        self.multiple_layers_widget.call_button.visible = False
        self.add_magic_function(self.multiple_layers_widget, image_layout)
        select_button = QPushButton('Select layers')
        select_button.clicked.connect(self.select_layers)
        image_layout.addWidget(select_button)
        
        self.choose_layer_widget = choose_layer()
        self.choose_layer_widget.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget, image_layout)
        select_button = QPushButton('Select image layer')
        select_button.clicked.connect(self.select_layer)
        image_layout.addWidget(select_button)
        
        self.choose_points_layer_widget = choose_points_layer()
        self.choose_points_layer_widget.call_button.visible = False
        self.add_magic_function(self.choose_points_layer_widget, image_layout)
        select_points_button = QPushButton('Select points layer')
        select_points_button.clicked.connect(self.select_points_layer)
        image_layout.addWidget(select_points_button)

        settings_layout = QVBoxLayout()
        add_section(settings_layout,'Settings')
        layout.addLayout(settings_layout)
        self.createSettings(settings_layout)
        
    def add_magic_function(self, widget, _layout):
        self.viewer.layers.events.inserted.connect(widget.reset_choices)
        self.viewer.layers.events.removed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)
            
    def select_layer(self,image: Image):
    
        image = self.choose_layer_widget.image.value
        
        if image.data.ndim == 3:
            
            self.imageRaw_name = image.name
            sz,sy,sx = image.data.shape
            # print(sz, sy, sx)
            # if not hasattr(self, 'h'): 
            #     self.start_preprocessing()
            print(f'Selected image: {image.name}')
    
    def select_layers(self, x: List[Layer]):
        
        names = []
        layers_list = self.example_widget.x.value
        
        for layer in layers_list:
            self._layer_name = layer.name
            names.append(layer.name)
            print(f'Selected layer: {layer.name}')
        print(names)
        return names
    
    def select_points_layer(self,       ##TODO it is not correct like this, we need one function that collects all layers
                     points: Points):
    
        points = self.choose_points_layer_widget.points.value
        self._points_name = points.name
        print(f'Selected points: {points.name}')

    def createSettings(self, slayout):
        
        layout = QVBoxLayout() ##TODO change layout, put the 2 checkboxes one near to the other
        
        self.inplace = Settings('Inplace operations', dtype=bool, initial=False,
                                      layout=slayout, 
                                      write_function = self.set_preprocessing)
        
        self.track = Settings('Track', dtype=bool, initial=False,
                                      layout=slayout,
                                      write_function = self.set_preprocessing)
        
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
        
    #     # self.listlayers = Combo_box(name = 'Layer_lit', choices = Accel.available(),
    #     #                           layout=right_layout,
    #     #                           write_function = self.setReconstructor)    
        # buttons
        buttons_dict = {'Select ROI': self.select_ROIs,
                        'Binning': self.bin_stack
                        }
        for button_name, call_function in buttons_dict.items():
            button = QPushButton(button_name)
            button.clicked.connect(call_function)
            slayout.addWidget(button)
        self.messageBox = QLineEdit()
        layout.addWidget(self.messageBox, stretch=True)
        self.messageBox.setText('Messages') #not working, I think
        
    
    def get_image(self):
        try:
            
            return self.viewer.layers[self.imageRaw_name].data
        except:
             raise(KeyError(r'Please select a valid 3D image ($\theta$, q, z)'))
             
    def get_points(self): ##TODO it would be good to have just one function to select all layers
        try:
            
            return self.viewer.layers[self._points_name].data
        except:
             raise(KeyError(r'Please select a valid Points layer'))
             
    def show_image(self, image_values, _inplace_value, fullname):
        
        # if 'scale' in kwargs.keys():    #GT: do we need to scale?
        #     scale = kwargs['scale']
        # else:
        #     scale = [1.]*image_values.ndim
        
        if _inplace_value == True:   ##TODO consider to add option to update existing layer
            fullname = self.imageRaw_name
            
            self.viewer.layers[fullname].data = image_values
            # self.viewer.layers[fullname].scale = scale
        
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = fullname,
                                            # scale = scale,
                                            interpolation2d = 'linear')
            return layer
    
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
    def select_ROIs(self): ##TODO add tracking option for select_ROIs and bin_stack
 
        original_image = self.get_image()
        points = self.get_points()
        _inplace_value = self.inplace.val
        fullname = 'ROI_' + self.imageRaw_name
        notifications.show_info(f'UL corner coordinates: {points[0][1:].astype(int)}')
        selected_roi, roi_pars = select_roi(original_image, points[-1], self.roi_height.val, self.roi_width.val)
        self.show_image(selected_roi, _inplace_value, fullname)

    #binning
    def bin_stack(self):
    
            if self.bin_factor != 1:
                original_image = self.get_image()
                _inplace_value = self.inplace.val
                fullname = 'Binned_' + self.imageRaw_name
                binned_roi = bin_3d(original_image, self.bin_factor.val)
                notifications.show_info(f'Original shape: {original_image.shape}, binned shape: {binned_roi.shape}')
                self.show_image(binned_roi, _inplace_value, fullname)
            else:
                notifications.show_info('Bin factor is 1, nothing to do.')
                
def current_layers(_):
    return list(current_viewer().layers)

@magic_factory(x={'widget_type': 'Select', "choices": current_layers})
def multiple_layers(x: List[Layer]):
    pass
        
@magic_factory
def choose_layer(image: Image):
        pass #TODO: substitute with a qtwidget without magic functions
@magic_factory
def choose_points_layer(points: Points):
        pass #TODO: substitute with a qtwidget without magic functions

if __name__ == '__main__':
    import napari
    
    viewer = napari.Viewer()
    mywidget = PreprocessingnWidget(viewer)
    viewer.window.add_dock_widget(mywidget, name = 'Preprocessing Widget')
    # warnings.filterwarnings('ignore')
    napari.run()