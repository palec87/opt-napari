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
from functools import partial

from napari.layers import Layer, Image, Points
from magicgui import magic_factory
from magicgui.widgets import create_widget
from napari.utils import notifications
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

ID_NAME = "_CLUSTER_ID"


def layer_container_and_selection(viewer=None):
    """
    Create a container and a dropdown widget to select the layer.

    Returns
    -------
    A tuple containing a QWidget for displaying the layer selection container,
    and a QWidget containing the selection options for the layer.
    """
    layer_selection_container = QWidget()
    layer_selection_container.setLayout(QHBoxLayout())
    layer_selection_container.layout().addWidget(QLabel("Layer"))
    layer_select = create_widget(annotation=Layer, label="layer")
    layer_selection_container.layout().addWidget(layer_select.native)

    if viewer is not None and viewer.layers.selection.active is not None:
        layer_select.value = viewer.layers.selection.active

    return layer_selection_container, layer_select


def update_properties_list(widget, exclude_list):
    """
    Updates the properties list of a given widget with the pr
    """
    pass


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

        (
            layer_selection_container,
            self.layer_select,
        ) = layer_container_and_selection(viewer=self.viewer)

        # update axes combo boxes automatically if features of
        # layer are changed
        self.last_connected = None
        self.layer_select.changed.connect(self.activate_property_autoupdate)
        
        image_layout = QSplitter(Qt.Vertical)#QVBoxLayout()
        layout.addWidget(image_layout)
        self.layout().addWidget(layer_selection_container)
        # add_section(image_layout, 'Image selection')
        # layout.addLayout(image_layout)
        
        # self.multiple_layers_widget = multiple_layers()
        # self.multiple_layers_widget.call_button.visible = False
        # self.add_magic_function(self.multiple_layers_widget, image_layout)
        # select_button = QPushButton('Select layers')
        # select_button.clicked.connect(self.select_layers)
        # image_layout.addWidget(select_button)
        
        # self.choose_layer_widget = choose_layer()
        # self.choose_layer_widget.call_button.visible = False
        # self.add_magic_function(self.choose_layer_widget, image_layout)
        # select_button = QPushButton('Select image layer')
        # select_button.clicked.connect(self.select_layer)
        # image_layout.addWidget(select_button)

        # self.selectImageLayerWidget = create_widget(annotation=Image,
        #                                             label="Image_layer",
        #                                             )
        # image_layout.addWidget(self.selectImageLayerWidget.native)
        # self.installEventFilter(self)

        # self.choose_points_layer_widget = choose_points_layer()
        # self.choose_points_layer_widget.call_button.visible = False
        # self.add_magic_function(self.choose_points_layer_widget, image_layout)
        # select_points_button = QPushButton('Select points layer')
        # select_points_button.clicked.connect(self.select_points_layer)
        # image_layout.addWidget(select_points_button)

        settings_layout = QVBoxLayout()
        layout.addLayout(settings_layout)
        self.createSettings(settings_layout)

    def activate_property_autoupdate(self):
        if self.last_connected is not None:
            self.last_connected.events.properties.disconnect(
                partial(update_properties_list, self, [ID_NAME])
            )
        self.layer_select.value.events.properties.connect(
            partial(update_properties_list, self, [ID_NAME])
        )
        self.last_connected = self.layer_select.value

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.layer_select.reset_choices(event)
            
    def select_layer(self, image: Image):

        image = self.choose_layer_widget.image.value

        if image.data.ndim == 3:

            self.imageRaw_name = image.name
            sz,sy,sx = image.data.shape
            # print(sz, sy, sx)
            # if not hasattr(self, 'h'): 
            #     self.start_preprocessing()
            print(f'Selected image: {image.name}')

    def select_layers(self, x: List[Layer]):
        print(x)
        names = []
        # layers_list = self.example_widget.x.value
        # layers_list = x

        for layer in x:
            # self._layer_name = layer.name
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
                
# def current_layers(_):
#     return list(current_viewer().layers)

# @magic_factory(x={'widget_type': 'Select', "choices": current_layers})
# def multiple_layers(x: List[Layer]):
#     pass
        
# @magic_factory
# def choose_layer(image: Image):
#         pass #TODO: substitute with a qtwidget without magic functions
# @magic_factory
# def choose_points_layer(points: Points):
#         pass #TODO: substitute with a qtwidget without magic functions

if __name__ == '__main__':
    import napari
    
    viewer = napari.Viewer()
    mywidget = PreprocessingnWidget(viewer)
    viewer.window.add_dock_widget(mywidget, name = 'Preprocessing Widget')
    # warnings.filterwarnings('ignore')
    napari.run()