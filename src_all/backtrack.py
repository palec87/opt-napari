import numpy as np
from napari.layers import Image
from napari.utils import notifications
from enum import Enum

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
        # print('debug update history', history.roi_def)
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
        # print('debug undo', history.roi_def)
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
