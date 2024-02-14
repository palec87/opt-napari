import numpy as np
from napari.layers import Image, Points
from magicgui import magic_factory
from napari import Viewer
from napari.utils import progress, notifications

from typing import Annotated
import warnings
from corrections import Correct
from enum import Enum

from utils import (
    select_roi,
    bin_3d,
)
from dataclasses import dataclass


class update_modes(Enum):
    inplace_track = (True, True)
    inplace_NOtrack = (True, False)
    NOinplace_NOtrack = (False, False)


@dataclass
class Backtrack:
    """
    Supports one undo operation. Only for inplace operations.
    Can break if:
        You operate on more datasets in parallel
    """
    raw_data: np.ndarray = None
    roi_def: tuple = ()  # indices refer always to raw data
    inplace: bool = True
    track: bool = False
    history_item = dict()

    #############################
    # This is all stupid and should be done as QT widget
    #############################
    def set_settings(self, inplace_value: bool, track_value: bool):
        """Global settings for inplace operations and tracking

        Args:
            inplace_value (bool): if operation are inplace (saving RAM)
            track_value (bool): Tracking to enable revertign inplace operations
        """
        self.inplace = inplace_value
        self.track = track_value

    def update_history(self, image: Image, data_dict: dict) -> np.ndarray:
        # for the first operation, store original as raw data
        if self.raw_data is None:
            self.raw_data = image.data

        # if no tracking, no update of history and return new image
        if not self._update_compatible():
            return data_dict['data']

        # not that necessary check
        if data_dict['operation'] not in ['roi', 'bin', 'correct']:
            raise AttributeError('Unknown operation to update history.')

        # compatible with update, I need to put old data here
        self.history_item['operation'] = data_dict['operation']
        self.history_item['data'] = image.data
        # for binning operation
        # this is not elegant at all
        try:
            self.history_item['roi_def'] = self.roi_def
            history.update_roi_pars(data_dict['roi_def'])
        except:
            pass

        # binning operation
        try:
            self.history_item['bin_factor'] = data_dict['bin_factor']
            # I need to update the roi pars too.
            history.update_roi_pars(data_dict['roi_def'])
        except:
            pass
        print('debug update history', history.roi_def)
        return data_dict['data']

    def _update_compatible(self) -> bool:
        """Will proceed to update only if inplace and tracking
        are True

        Returns:
            bool: if history update is going to run.
        """
        if self.inplace and self.track:
            return True
        else:
            return False

    def undo(self):
        if self.history_item == dict():
            raise ValueError('No State to revert to.')

        if self.history_item['operation'] == 'roi':
            notifications.show_info('Reverting ROI selection')
            self.roi_def = self.history_item['roi_def']

        elif self.history_item['operation'] == 'bin':
            # what about binning of the correction files?
            # nothing here I think
            notifications.show_info('Reverting binning.')

        elif self.history_item['operation'] == 'correct':
            notifications.show_info('Reverting corrections.')

        else:
            raise ValueError('Unsupported operation')
        # resetting history dictionary
        data = self.history_item.pop('data')
        self.history_item = dict()
        print('debug undo', history.roi_def)
        return data

    def revert_to_raw(self):
        self.history_item = dict()
        return self.raw_data

    def update_roi_pars(self, roi_pars):
        print('updating roi_params, should run only once.')
        if self.roi_def == ():
            self.roi_def = roi_pars
        else:
            # ULy, height, ULx, width
            i1, i2, i3, i4 = self.roi_def
            j1, j2, j3, j4 = roi_pars
            self.roi_def = (
                i1 + j1, i1 + j1 + j2,
                i3 + j3, i3 + j3 + j4,
            )


history = Backtrack()


# This widget just because not using pyqt
@magic_factory(call_button="Update",
               Operation_mode={"choices": update_modes})
def settings(Operation_mode=update_modes.inplace_NOtrack):
    """Widget to select global settings of inplace operations and tracking

    Args:
        Operation_mode (enum, optional): Option mode. 
            Defaults to update_modes.inplace_notrack.
    """
    history.set_settings(*Operation_mode.value)


# ROI
@magic_factory(call_button="Select ROI")
def select_ROIs(viewer: Viewer,
                image: Image,
                points_layer: Points,
                roi_height: Annotated[int, {'min': 1, 'max': 5000}] = 200,
                roi_width: Annotated[int, {'min': 1, 'max': 5000}] = 200,
                ):
    original_stack = np.asarray(image.data)
    points = np.asarray(points_layer.data)
    notifications.show_info(f'UL corner coordinates: {points[0][1:].astype(int)}')

    selected_roi, roi_pars = select_roi(original_stack,
                                        points[-1],
                                        roi_height,
                                        roi_width)
    if history.inplace:
        new_data = {'operation': 'roi',
                    'data': selected_roi,
                    'roi_def': roi_pars}
        image.data = history.update_history(image, new_data)
    else:
        viewer.add_image(selected_roi)
    print('debug select ROI', history.roi_def)


# Revert history
@magic_factory(call_button="Undo Last")
def revert_last(viewer: Viewer, image: Image):
    """This is always inpalce operation

    Args:
        viewer (Viewer): napari viewer
        image (Image): image layer
    """
    try:
        image.data = history.undo()
    except Exception as e:
        print(e)


# Binning
# TODO: need to fix the nondivisable binning
@magic_factory(call_button="Bin")
def bin_stack(viewer: Viewer,
              image: Image,
              bin_factor: Annotated[int, {'min': 1, 'max': 500}] = 2,
              ):
    # binning, only if bin_factor is not 1
    if bin_factor != 1:
        binned_roi = bin_3d(image.data, bin_factor)
        notifications.show_info(f'Original shape: {image.data.shape}, binned shape: {binned_roi.shape}')
        # viewer.add_image(binned_roi)
    else:
        notifications.show_info('Bin factor is 1, nothing to do.')
        return

    if history.inplace:
        if history.roi_def == ():
            new_roi = (0, binned_roi.shape[-2],
                       0, binned_roi.shape[-1])
        else:
            new_roi = (history.roi_def[0], binned_roi.shape[-2],
                       history.roi_def[1], binned_roi.shape[-1],
                       )

        new_data = {'operation': 'bin',
                    'data': binned_roi,
                    'factor': bin_factor,
                    'roi_def': new_roi}

        image.data = history.update_history(image, new_data)
    else:
        viewer.add_image(binned_roi)
    print('debug bin stack', history.roi_def)


# Image corrections
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
    corr = Correct(hot, std_cutoff, dark, bright)
    data_corr = np.zeros(original_stack.shape,
                         dtype=original_stack.dtype)
    print(f'number of hot pixels: {len(corr.hot_pxs)}')

    if dark is not None:
        for i, img in progress(enumerate(original_stack)):
            data_corr[i] = corr.correct_dark(img)
        print(f'{np.amax(data_corr)}, min: {np.amin(data_corr)}')
        notifications.show_info('Dark correction done.')

    if bright is not None:
        for i, img in progress(enumerate(data_corr)):
            data_corr[i] = corr.correct_bright(img)
        notifications.show_info('Bright correction done.')

    if hot is not None:
        print('to be implemented')

    # now updating the image.
    if history.inplace:
        new_data = {'operation': 'correct',
                    'data': data_corr,
                    }
        image.data = history.update_history(image, new_data)
    else:
        viewer.add_image(data_corr)
    print('debug corrections', history.roi_def)


if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    roi_widget = select_ROIs()
    correct_widget = corrections()
    bin_widget = bin_stack()
    revert_widget = revert_last()
    settings_widget = settings()

    viewer.window.add_dock_widget(settings_widget, name='Settings',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(revert_widget, name='Undo Last',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(correct_widget, name='Image Corrections',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(roi_widget, name='select ROI',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(bin_widget, name='Bin Stack',
                                  area='right', add_vertical_stretch=True)
    warnings.filterwarnings('ignore')
    napari.run()
