import numpy as np
from napari.layers import Image, Points
from magicgui import magic_factory
from napari import Viewer
from napari.utils import notifications

from typing import Annotated
import warnings
import os 
from .widget_settings import Settings, Combo_box
#import processors
import napari
from qtpy.QtWidgets import QVBoxLayout, QSplitter, QHBoxLayout, QWidget, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QLabel
# from qtpy.QtCore import Qt
from napari.layers import Image
import numpy as np
from napari.qt.threading import thread_worker
import warnings
from time import time
import scipy.ndimage as ndi
from enum  import Enum
import cv2 
import tqdm
from corrections import Correct

from utils import (
    select_roi,
    bin_3d,
)
from dataclasses import dataclass

class PreprocessingnWidget(QWidget):
    name = 'Preprocessor'
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.setup_ui()
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
        
        self.choose_layer_widget = choose_layer()
        self.choose_layer_widget.call_button.visible = False
        self.add_magic_function(self.choose_layer_widget, image_layout)
        select_button = QPushButton('Select image layer')
        select_button.clicked.connect(self.select_layer)
        image_layout.addWidget(select_button)

        settings_layout = QVBoxLayout()
        add_section(settings_layout,'Settings')
        layout.addLayout(settings_layout)
        self.createSettings(settings_layout)

    def createSettings(self, slayout):
        
        self.reshapebox = Settings('Reshape volume',
                                  dtype=bool,
                                  initial = True, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)        

        self.resizebox = Settings('Reconstruction size',
                                  dtype=int, 
                                  initial=100, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)
        
        self.clipcirclebox = Settings('Clip to circle',
                            dtype=bool,
                            initial = False, 
                            layout=slayout, 
                            write_function = self.set_opt_processor)        

        self.filterbox = Settings('Use filtering',
                            dtype=bool,
                            initial = False, 
                            layout=slayout, 
                            write_function = self.set_opt_processor) 
        

        self.registerbox = Settings('Automatic axis alignment',
                                  dtype=bool,
                                  initial=False, 
                                  layout=slayout, 
                                  write_function = self.set_opt_processor)
        
        self.manualalignbox = Settings('Manual axis alignment',
                            dtype=bool,
                            initial = False, 
                            layout=slayout, 
                            write_function = self.set_opt_processor) 
        
        self.alignbox = Settings('Axis shift',
                            dtype=int,
                            vmin=-500,
                            vmax=500, 
                            initial=0, 
                            layout=slayout, 
                            write_function = self.set_opt_processor)
        
        
        
        #create combobox for reconstruction method
        self.reconbox = Combo_box(name ='Reconstruction method',
                             initial = Rec_modes.FBP_GPU.value,
                             choices = Rec_modes,
                             layout = slayout,
                             write_function = self.set_opt_processor)
        
        self.orderbox = Combo_box(name ='Rotation axis',
                             initial = Order_Modes.Horizontal.value,
                             choices = Order_Modes,
                             layout = slayout,
                             write_function = self.set_opt_processor)

        # add calculate psf button
        calculate_btn = QPushButton('Reconstruct')
        calculate_btn.clicked.connect(self.stack_reconstruction)
        slayout.addWidget(calculate_btn)


@dataclass
class Backtrack:
    """ 
    Supports one undo operation. Only for inplace operations.
    Will break if:
        You operate on more datasets in parallel
    """
    # def __init__(self) -> None:
    raw_data: np.ndarray = None
    last_data: np.ndarray = None
    # roi_def: tuple = ()  # indices refer always to raw data
    track: bool = False
    inplace: bool = False
    history_item = dict()

    def update_tracking(self, value: bool):
        
        self.track = value
        if self.track is True:
            pass


    def update_history(self, operation, data_dict):
        if operation == 'roi':
            pass
        elif operation == 'bin':
            pass
        elif operation == 'correct':
            pass
        else:
            raise TypeError('Unknown operation to update history.')
        
        self.operation = operation
        
    def revert_history(self):
        pass

    def revert_to_raw(self):
        pass


history = Backtrack()


# ROI
@magic_factory(call_button="Select ROI")
def select_ROIs(viewer: Viewer,
                image: Image,
                points_layer: Points,
                roi_height: Annotated[int, {'min': 1, 'max': 5000}] = 200,
                roi_width: Annotated[int, {'min': 1, 'max': 5000}] = 200,
                inplace: bool = True,
                track_history: bool = False,
                ):
    shared_variables.track = track_history
    original_stack = np.asarray(image.data)
    points = np.asarray(points_layer.data)
    notifications.show_info(f'UL corner coordinates: {points[0][1:].astype(int)}')

    selected_roi, roi_pars = select_roi(original_stack,
                                        points[-1],
                                        roi_height,
                                        roi_width)
    # if I do this twice, I will get wrong indicis for corrections!!!
    update_roi_pars(roi_pars)
    # shared_variables.roi_def: roi_pars
    # partial solution to tracking history
    # TODO: does not work, if you are on wrong layer, or perhaps
    # user deleted one layer going back one step, and wants to do another step
    # Or changes the inplace parameter inbetween
    # LOts of lose ends here!!!!
    if inplace:
        update_history(inplace, image, selected_roi)
    else:
        viewer.add_image(selected_roi)


# this will not work, if user applies it once on the roi and then again on the
# raw, fuck me...
# Fixing means making the history a proper class.
def update_roi_pars(roi_pars):
    if shared_variables.roi_def == ():
        shared_variables.roi_def = roi_pars
    else:
        i1, i2, i3, i4 = shared_variables.roi_def
        j1, j2, j3, j4 = roi_pars
        shared_variables.roi_def = (
            i1 + j1, i1 + j1 + j2,
            i3 + j3, i3 + j3 + j4,
        )


def update_history(inplace: bool, image: Image, new_image: np.ndarray):
    if not shared_variables.track:
        notifications.show_info('Tracking not enabled, no history to update (saving RAM).')
        return
    if not inplace:
        notifications.show_info('Not an inplace operation, no tracking, just delete the layer.')
        return

    if shared_variables.raw_data is None:
        shared_variables.raw_data = image.data.copy()
    else:
        shared_variables.last_data = image.data.copy()

    image.data = new_image
    return


# Revert history
@magic_factory(call_button="Undo Last")
def revert_last(viewer: Viewer, image: Image, inplace: bool):
    if not shared_variables.track:
        notifications.show_info('Tracking not enabled, no history to update (saving RAM).')
        return

    if shared_variables.last_data is not None:
        notifications.show_info('Reverting to last data')
        if inplace:
            image.data = shared_variables.last_data
        else:
            viewer.add_image(shared_variables.last_data)
        shared_variables.last_data = None

    elif shared_variables.last_data is None and shared_variables.raw_data is not None:
        notifications.show_info('Reverting to raw data')
        if inplace:
            image.data = shared_variables.raw_data
        else:
            viewer.add_image(shared_variables.raw_data)
    else:
        notifications.show_info('Nothing to revert to.')


# Binning
@magic_factory(call_button="Bin")
def bin_stack(viewer: Viewer,
              image: Image,
              bin_factor: Annotated[int, {'min': 1, 'max': 500}] = 2,
              ):
    if bin_factor != 1:
        binned_roi = bin_3d(image.data, bin_factor)
        notifications.show_info(f'Original shape: {image.data.shape}, binned shape: {binned_roi.shape}')
        viewer.add_image(binned_roi)
    else:
        notifications.show_info('Bin factor is 1, nothing to do.')


# Image corrections
# use Correct class
@magic_factory(call_button="Image Corrections")
def corrections(viewer: Viewer,
                image: Image,
                hot: Image = None,
                std_cutoff: int = 5,
                dark: Image = None,
                bright: Image = None,
                ):
    original_stack = np.asarray(image.data)
    # init correction class
    Correct(hot, std_cutoff, dark, bright)

    # # first correct bright field for dark
    # bright_corr = bright.data - dark.data

    # ans = np.empty(original_stack.shape)
    # for i, img in enumerate(original_stack):
    #     ans[i] = img - dark.data - bright_corr

    # viewer.add_image(ans)

@magic_factory
def choose_layer(image: Image):
        pass #TODO: substitute with a qtwidget without magic functions

if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    roi_widget = select_ROIs()
    correct_widget = corrections()
    bin_widget = bin_stack()
    revert_widget = revert_last()

    viewer.window.add_dock_widget(revert_widget, name='Undo Last',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(roi_widget, name='select ROI',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(bin_widget, name='Bin Stack',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(correct_widget, name='Image Corrections',
                                  area='right', add_vertical_stretch=True)
    warnings.filterwarnings('ignore')
    napari.run()
