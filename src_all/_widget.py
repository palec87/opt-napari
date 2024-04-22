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
    inplace_track = (True, True)        # record history of last operation
    inplace_NOtrack = (True, False)     # Inplace operations without tracking (dangerous)
    NOinplace_NOtrack = (False, False)  # Creating copies no tracking needed


@dataclass
class Backtrack:
    """
    Supports one undo operation. Only for inplace operations.
    Can break if:
        You operate on more datasets in parallel
    """
    raw_data: np.ndarray = None
    roi_def: tuple = ()     # indices refer always to raw data
    history_item = dict()   # operation, data, roi_def, bin_factor

    # DP: The flags will be moved to the widget I think
    inplace: bool = True
    track: bool = False

    #############################
    # DP: parts needs to stay, flags from checkbox updates can control behavior
    #############################

    # DP this will be changed
    def set_settings(self, inplace_value: bool, track_value: bool):
        """Global settings for inplace operations and tracking

        Args:
            inplace_value (bool): if operation are inplace (saving RAM)
            track_value (bool): Track enables reverting the inplace operations
        """
        self.inplace = inplace_value
        self.track = track_value

    def update_history(self, image: Image, data_dict: dict) -> np.ndarray:
        """Updates history if trackiing and inplace operations are selected.

        Args:
            image (Image): napari image .data atribute is the np.ndarray image
            data_dict (dict): metadata and data for the operation to register

        Raises:
            AttributeError: In case unknown operation passed

        Returns:
            np.ndarray: new image, which is from the Image
        """
        # for the first operation, store original as raw data
        if self.raw_data is None:
            self.raw_data = image.data

        # if no tracking, no update of history and return new image
        if not self._update_compatible():
            return data_dict['data']

        # DP: not that necessary check, can be removed I think
        if data_dict['operation'] not in ['roi', 'bin', 'correct']:
            raise AttributeError('Unknown operation to update history.')

        # compatible with update, I put old data to history item
        # and update current parameters in 
        self.history_item['operation'] = data_dict['operation']
        self.history_item['data'] = image.data
        # for binning operation
        # TODO: this is not elegant at all
        try:
            self.history_item['roi_def'] = self.roi_def
            self.update_roi_pars(data_dict['roi_def'])
        except:
            pass

        # binning operation
        try:
            self.history_item['bin_factor'] = data_dict['bin_factor']
            # I need to update the roi pars too.
            self.update_roi_pars(data_dict['roi_def'])
        except:
            pass
        print('debug update history', history.roi_def)
        return data_dict['data']

    # DP, this should be checked upon Qt widget values
    def _update_compatible(self) -> bool:
        """Will proceed to update only if inplace and tracking
        are True.

        Returns:
            bool: if history update is going to run.
        """
        if self.inplace and self.track:
            return True
        else:
            return False

    def undo(self) -> np.ndarray:
        """Performs the actual undo operation. If history item
        exists, it identifies, which operation needs to be reverted
        to update the parameters. Image data are updated from the
        history dictionary too.

        Raises:
            ValueError: No history item to revert to.
            ValueError: Unsupported operation in the history

        Returns:
            np.ndarray: reverted image data
        """
        if self.history_item == dict():
            raise ValueError('No State to revert to.')

        if self.history_item['operation'] == 'roi':
            notifications.show_info('Reverting ROI selection')
            self.roi_def = self.history_item['roi_def']

        elif self.history_item['operation'] == 'bin':
            # what about binning of the correction files?
            # TODO: corrections need to run before binning. Tha is necessary for the bad pixels
            # but not for dark and bright field. Could be fixed, or need to be imposed or at least
            # raised as warnings
            notifications.show_info('Reverting binning.')

        elif self.history_item['operation'] == 'correct':
            notifications.show_info('Reverting corrections.')

        else:
            raise ValueError('Unsupported operation')

        # resetting history dictionary, because only 1 operation can be tracked
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
            # TODO: DP This indexing is awful
            # ULy, height, ULx, width
            i1, i2, i3, i4 = self.roi_def
            j1, j2, j3, j4 = roi_pars
            self.roi_def = (
                i1 + j1, i1 + j1 + j2,
                i3 + j3, i3 + j3 + j4,
            )


# DP, perhaps we keep the history in separate class even for the QT widget.
history = Backtrack()


# This widget exists just because not using pyqt
# Checkbox in Qt solves this
@magic_factory(call_button="Update",
               Operation_mode={"choices": update_modes})
def settings(Operation_mode=update_modes.inplace_NOtrack):
    """Widget to select global settings of inplace operations and tracking

    Args:
        Operation_mode (enum, optional): Option mode. 
            Defaults to update_modes.inplace_notrack.
    """
    history.set_settings(*Operation_mode.value)


# ROI, this will have a button to run
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
    """This is always inplace operation

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
              ) -> None:
    """
    Binning of the 3D stack along along the axis 1 and 2. Axis 0
    is the image stacking dimension. Mean binning is performed to avoid
    overflows.

    Args:
        viewer (Viewer): napari viewer
        image (Image): napari Image
        bin_factor (int, optional): square bin factor, mean of bin_factor**2
            results in a new pixel value. Defaults to 2.
    """
    # binning, only if bin_factor is not 1
    if bin_factor != 1:
        binned_roi = bin_3d(image.data, bin_factor)
        notifications.show_info(f'Original shape: {image.data.shape}, binned shape: {binned_roi.shape}')
    else:
        notifications.show_info('Bin factor is 1, nothing to do.')
        return

    # DP: move to separate func or not?
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
# QT widget allows to do corrections individually, instead of one button
# for all which is better for the user
@magic_factory(call_button="Image Corrections")
def corrections(viewer: Viewer,
                image: Image,
                hot: Image = None,
                std_cutoff: int = 5,
                dark: Image = None,
                bright: Image = None,
                ) -> None:
    """Performs all the corerctions if corerction images are selected.
    If not, correction is skipped. Like this, do correction one by one
    implies lots of clicking, better to have separate buttons for each
    correction

    Args:
        viewer (Viewer): napari viewer
        image (Image): napari image
        hot (Image, optional): image layer with hot pixels acquisition.
            Defaults to None.
        std_cutoff (int, optional): pixels which are further from the
            mean than STD*cutoff are identified as bad (HOT) pixels.
            This is camera dependent. Defaults to 5.
        dark (Image, optional): image layer containing dark-field image.
            Defaults to None.
        bright (Image, optional): image layer containing brright-filed image.
            Defaults to None.
    """
    original_stack = np.asarray(image.data)
    # init correction class
    corr = Correct(hot, std_cutoff, dark, bright)

    # preallocate corrected array
    data_corr = np.zeros(original_stack.shape,
                         dtype=original_stack.dtype)
    # print(f'number of hot pixels: {len(corr.hot_pxs)}')

    # dark-field
    if dark is not None:
        for i, img in progress(enumerate(original_stack)):
            data_corr[i] = corr.correct_dark(img)
            print(f'max: {np.amax(data_corr)}, min: {np.amin(data_corr)}')
        notifications.show_info('Dark correction done.')

    # bright-field
    if bright is not None:
        for i, img in progress(enumerate(data_corr)):
            data_corr[i] = corr.correct_bright(img)
        notifications.show_info('Bright correction done.')

    # Bad pixels
    if hot is not None:
        print('to be implemented')

    # TODO: intensity correction not implemented!!!
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

    # widgets
    roi_widget = select_ROIs()
    correct_widget = corrections()
    bin_widget = bin_stack()
    revert_widget = revert_last()
    settings_widget = settings()

    # docking
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
    # warnings.filterwarnings('ignore')
    napari.run()
