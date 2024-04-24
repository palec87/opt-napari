import numpy as np

import napari
from napari.layers import Image, Points
from napari.utils import progress, notifications
from magicgui.widgets import create_widget

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLineEdit,
    QRadioButton, QLabel, QFrame,
    QButtonGroup, QGroupBox,
)

from widget_settings import Settings
from corrections import Correct
from backtrack import Backtrack
from utils import (
    select_roi,
    bin_3d,
)

DEBUG = True
badPxDict = {
    'Identify Hot pxs': 'hot',
    'Identify Dead pxs': 'dead',
    'Identify Both': 'both',
}


def layer_container_and_selection(
        viewer=None,
        layer_type=Image,
        container_name='Layer'):
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
    layer_select = create_widget(annotation=layer_type, label="layer")
    layer_selection_container.layout().addWidget(layer_select.native)
    layer_selection_container.layout().setContentsMargins(0, 0, 0, 0)

    if viewer is not None and viewer.layers.selection.active is not None:
        layer_select.value = viewer.layers.selection.active

    return layer_selection_container, layer_select


class PreprocessingnWidget(QWidget):
    name = 'Preprocessor'

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.corr = None
        self.history = Backtrack()
        self.setup_ui()

    def correctDarkBright(self):
        # display message in the message box
        self.messageBox.setText('Correcting Dark and Bright')

        # TODO: inplace and tracking needs to be taken care of here.

        original_image = self.image_layer_select.value
        self.data_corr = np.zeros(original_image.data.shape,
                                  dtype=original_image.data.dtype)

        self.corr = Correct(dark=self.dark_layer_select.value.data,
                            bright=self.bright_layer_select.value.data,
                            )
        if self.flagBright.val:
            if self.flagDark.val and self.flagExp == 'Transmission':
                for i, img in progress(enumerate(original_image.data)):
                    self.data_corr[i] = ((img - self.corr.dark) /
                                         (self.corr.bright - self.corr.dark))
                # because it is float between 0-1, needs to be rescaled and casted to uint16
                # check if the image has element greater than 1 or less than 0
                if np.amax(self.data_corr) > 1 or np.amin(self.data_corr) < 0:
                    self.messageBox.setText(
                        'Image values out of range. Clipping to 0-1.')
                self.data_corr = (np.clip(self.data_corr, 0, 1) * 65535).astype(np.uint16)

            elif self.flagExp == 'Emmision':
                # this is integer, no casting needed
                for i, img in progress(enumerate(original_image.data)):
                    self.data_corr[i] = img - self.corr.bright
            else:  # transmission, no dark correction
                for i, img in progress(enumerate(original_image.data)):
                    self.data_corr[i] = img / self.corr.bright

                # because it is float between 0-1, needs to be rescaled and casted to uint16
                # make sure that the image is between 0-1 first
                if np.amax(self.data_corr) > 1 or np.amin(self.data_corr) < 0:
                    self.messageBox.setText(
                        'Image values out of range. Clipping to 0-1.',
                        )
                self.data_corr = (np.clip(self.data_corr, 0, 1) * 65535).astype(np.uint16)
        else:  # only dark correction
            # this is integer, no casting needed
            for i, img in progress(enumerate(original_image.data)):
                self.data_corr[i] = img - self.corr.dark

        self.show_image(self.data_corr,
                        'Intensity correction' + original_image.name,
                        original_image.contrast_limits)

    # hot pixels correction (NOT WORKING YET)
    def correctBadPixels(self):
        original_image = self.image_layer_select.value

        # init correction class
        self.messageBox.setText('Correcting hot pixels.')
        corr = Correct(hot=self.hot_layer_select.value.data,
                       std_mult=self.std_cutoff.val)
        hotPxs, deadPxs = self.corr.get_bad_pxs(mode=self.flagBad)

        self.messageBox.setText(f'Number of hot pixels: {len(corr.hot_pxs)}')

        # Here should raise a yes/no dialog to ask if the user wants to correct the hot pixels
        # if yes, then the correction is done, if not, the hot pixels are displayed
        data_corr = np.zeros(original_image.data.shape,
                             dtype=original_image.data.dtype)
        for i, img in progress(enumerate(original_image.data)):
            data_corr[i] = corr.correctBadPxs(img)

        # self.show_image(selected_roi, fullname)
        self.show_image(data_corr,
                        'Hot correction' + original_image.name,
                        original_image.contrast_limits)

    def correctIntensity(self):
        self.messageBox.setText('Correcting intensity.')
        if self.corr is None:
            self.corr = Correct(bright=self.bright_layer_select.value.data)

        # data to correct
        original_image = self.image_layer_select.value
        data_corr = np.zeros(original_image.data.shape,
                             dtype=original_image.data.dtype)

        # intensity correction
        data_corr = self.corr.correct_int(original_image, use_bright=False,
                                          rect_dim=self.rectSize.val)
        self.show_image(data_corr,
                        'Intensity correction' + original_image.name,
                        original_image.contrast_limits)

    def select_ROIs(self):
        original_image = self.image_layer_select.value
        points = self.points_layer_select.value.data
        self.messageBox.setText(
            f'UL corner coordinates: {points[0][1:].astype(int)}')
        # notifications.show_info(f'UL corner coordinates: {points[0][1:].astype(int)}')

        # DP: do I need roi pars for tracking?
        selected_roi, roi_pars = select_roi(original_image.data,
                                            points[-1],  # last point
                                            self.roi_height.val,
                                            self.roi_width.val)
        self.show_image(selected_roi,
                        'ROI_' + original_image.name,
                        original_image.contrast_limits)

    def bin_stack(self):
        if self.bin_factor != 1:
            original_image = self.image_layer_select.value
            binned_roi = bin_3d(original_image.data, self.bin_factor.val)
            notifications.show_info(
                f'Original shape: {original_image.data.shape},'
                f'binned shape: {binned_roi.shape}')
            self.show_image(binned_roi,
                            'binned_' + original_image.name,
                            original_image.contrast_limits)
        else:
            notifications.show_info('Bin factor is 1, nothing to do.')

    def calcLog(self):
        self.messageBox.setText('Calculating -Log')
        original_image = self.image_layer_select.value
        data_corr = -np.log10(original_image.data)

        self.show_image(data_corr,
                        '-Log_' + original_image.name,
                        )

    ##################
    # Helper methods #
    ##################
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.image_layer_select.reset_choices(event)
        self.hot_layer_select.reset_choices(event)
        self.dark_layer_select.reset_choices(event)
        self.bright_layer_select.reset_choices(event)
        self.points_layer_select.reset_choices(event)

    def show_image(self, image_values, fullname, contrast=None):
        if self.inplace.val:
            fullname = self.image_layer_select.value.name
            self.viewer.layers[fullname].data = image_values
        else:
            self.viewer.add_image(image_values,
                                  name=fullname,
                                  contrast_limits=contrast,
                                  interpolation2d='linear')

    def set_preprocessing(self, *args):
        """ Sets preprocessing parameters
        """
        if hasattr(self, 'h'):
            self.h.inplace_val = self.inplace.val
            self.h.track_val = self.track.val
            self.h.cutoff_val = self.std_cutoff.val
            self.h.height_val = self.roi_height.val
            self.h.width_val = self.roi_width.val
            self.h.binning_val = self.bin_factor.val
            # self.h.set_reconstruction_process()

    ##############
    # UI methods #
    ##############
    def setup_ui(self):
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        mode_layout = QVBoxLayout()
        layout.addLayout(mode_layout)
        self.selectProcessingMode(mode_layout)

        # layers selection containers
        (image_selection_container,
         self.image_layer_select) = layer_container_and_selection(
                                        viewer=self.viewer,
                                        layer_type=Image,
                                        container_name='Image to analyze',
                                        )

        (hot_image_selection_container,
         self.hot_layer_select) = layer_container_and_selection(
                                    viewer=self.viewer,
                                    layer_type=Image,
                                    container_name='Hot correction image',
                                    )
        (dark_image_selection_container,
         self.dark_layer_select) = layer_container_and_selection(
                                        viewer=self.viewer,
                                        layer_type=Image,
                                        container_name='Dark correction image',
                                        )
        (bright_image_selection_container,
         self.bright_layer_select) = layer_container_and_selection(
                                        viewer=self.viewer,
                                        layer_type=Image,
                                        container_name='Bright correction image',
                                        )
        (points_selection_container,
         self.points_layer_select) = layer_container_and_selection(
                                        viewer=self.viewer,
                                        layer_type=Points,
                                        container_name='Points layer for ROI selection',
                                        )

        # layers selection layout
        image_layout = QVBoxLayout()
        layout.addLayout(image_layout)
        self.layout().addWidget(image_selection_container)
        self.layout().addWidget(hot_image_selection_container)
        self.layout().addWidget(dark_image_selection_container)
        self.layout().addWidget(bright_image_selection_container)
        self.layout().addWidget(points_selection_container)

        # inplace and track options
        settings_layout = QVBoxLayout()
        layout.addLayout(settings_layout)

        self.createSettings(settings_layout)

    def selectProcessingMode(self, slayout):
        groupbox = QGroupBox('Global settings')
        box = QVBoxLayout()
        groupbox.setLayout(box)
        self.inplace = Settings('Inplace operations',
                                dtype=bool,
                                initial=False,
                                layout=box,
                                write_function=self.set_preprocessing)

        self.track = Settings('Track',
                              dtype=bool,
                              initial=False,
                              layout=box,
                              write_function=self.set_preprocessing)

        # make self.track visible only when self.inplace is True
        slayout.addWidget(groupbox)
        self.setTrackVisibility()

    def setTrackVisibility(self):
        # initial setting
        self.track.sbox.setVisible(self.inplace.val)
        self.track.lab.setVisible(self.inplace.val)

        # update visibility of track box when inplace box is changed
        self.inplace.sbox.stateChanged.connect(
            lambda: self.track.sbox.setVisible(self.inplace.val))
        # update the label visibility of track box when inplace box is changed
        self.inplace.sbox.stateChanged.connect(
            lambda: self.track.lab.setVisible(self.inplace.val))

    def onExperimentChange(self, button):
        self.flagExp = button.text()
        self.messageBox.setText(f'Experiment modality: {self.flagExp}')

    def onBadPixelsChange(self, button):
        self.flagBad = badPxDict[button.text()]
        self.messageBox.setText(f'Bad pixels option: {self.flagBad}')

    def createSettings(self, slayout):
        # Linear structure to impose correct order

        groupbox1 = QGroupBox('Dark/Bright Correction')
        box1 = QVBoxLayout()
        groupbox1.setLayout(box1)

        # create QradioButtons
        groupExpMode = QButtonGroup(self)
        groupExpMode.buttonClicked.connect(self.onExperimentChange)

        transExp = QRadioButton('Transmission')
        transExp.setChecked(True)
        groupExpMode.addButton(transExp)
        box1.addWidget(transExp)

        emlExp = QRadioButton('Emmision')
        groupExpMode.addButton(emlExp)
        box1.addWidget(emlExp)

        # checkboxes
        self.flagDark = Settings('Include Dark',
                                 dtype=bool,
                                 initial=False,
                                 layout=box1,
                                 write_function=self.set_preprocessing)
        self.flagBright = Settings('Include Bright',
                                   dtype=bool,
                                   initial=False,
                                   layout=box1,
                                   write_function=self.set_preprocessing)
        # Now comes button
        self.addButton(box1, 'Correct Dark+Bringht', self.correctDarkBright)
        slayout.addWidget(groupbox1)

        # create a groupbox for hot pixel correction
        groupbox2 = QGroupBox('Bad pixels correction')
        box2 = QVBoxLayout()
        groupbox2.setLayout(box2)

        # Hot pixel correction
        self.std_cutoff = Settings('Hot STD cutoff',
                                   dtype=int,
                                   initial=5,
                                   vmin=1,
                                   vmax=20,
                                   layout=box2,
                                   write_function=self.set_preprocessing)

        groupBadPxMode = QButtonGroup(self)
        groupBadPxMode.buttonClicked.connect(self.onBadPixelsChange)

        flagGetHot = QRadioButton('Identify Hot pxs')
        flagGetHot.setChecked(True)
        groupBadPxMode.addButton(flagGetHot)
        box2.addWidget(flagGetHot)

        flagGetDead = QRadioButton('Identify Dead pxs')
        groupBadPxMode.addButton(flagGetDead)
        box2.addWidget(flagGetDead)

        flagGetBoth = QRadioButton('Identify Both')
        groupBadPxMode.addButton(flagGetBoth)
        box2.addWidget(flagGetBoth)

        self.addButton(box2, 'Hot pixel correction', self.correctBadPixels)
        slayout.addWidget(groupbox2)

        # Intensity correction
        # create a groupbox for Intensity correction
        groupboxInt = QGroupBox('Intensity correction')
        boxInt = QVBoxLayout()
        groupboxInt.setLayout(boxInt)
        self.rectSize = Settings('Rectangle size',
                                 dtype=int,
                                 initial=50,
                                 vmin=10,
                                 vmax=500,
                                 layout=boxInt,
                                 write_function=self.set_preprocessing)
        self.addButton(boxInt, 'Intensity correction', self.correctIntensity)
        slayout.addWidget(groupboxInt)

        # select ROI
        groupboxRoi = QGroupBox('ROI selection')
        boxRoi = QVBoxLayout()
        groupboxRoi.setLayout(boxRoi)
        self.roi_height = Settings('ROI height',
                                   dtype=int,
                                   initial=200,
                                   vmin=1,
                                   vmax=5000,
                                   layout=boxRoi,
                                   write_function=self.set_preprocessing)

        self.roi_width = Settings('ROI width',
                                  dtype=int,
                                  initial=200,
                                  vmin=1,
                                  vmax=5000,
                                  layout=boxRoi,
                                  write_function=self.set_preprocessing)
        self.addButton(boxRoi, 'Select ROI', self.select_ROIs)
        slayout.addWidget(groupboxRoi)

        # binning
        groupboxBin = QGroupBox('Binning')
        boxBin = QVBoxLayout()
        groupboxBin.setLayout(boxBin)

        self.bin_factor = Settings('Bin factor',
                                   dtype=int,
                                   initial=2,
                                   vmin=1,
                                   vmax=500,
                                   layout=boxBin,
                                   write_function=self.set_preprocessing)
        self.addButton(boxBin, 'Bin Stack', self.bin_stack)
        slayout.addWidget(groupboxBin)

        # -Log
        self.addButton(slayout, '-Log', self.calcLog)

        # message box
        self.messageBox = QLineEdit()
        self.messageBox.setReadOnly(True)
        slayout.addWidget(self.messageBox, stretch=True)
        self.messageBox.setText('Messages')

    def addButton(self, layout, button_name, call_function):
        button = QPushButton(button_name)
        button.clicked.connect(call_function)
        layout.addWidget(button)


if __name__ == '__main__':
    import napari

    viewer = napari.Viewer()
    optWidget = PreprocessingnWidget(viewer)
    viewer.window.add_dock_widget(optWidget, name='OPT Preprocessing')
    # warnings.filterwarnings('ignore')
    if DEBUG:
        import glob
        # load example data from data folder
        viewer.open('src_all/sample_data/corr_hot.tiff', name='hot')
        viewer.open('src_all/sample_data/dark_field.tiff', name='dark')
        viewer.open('src_all/sample_data/flat_field.tiff', name='bright')

        # open OPT stack
        viewer.open(glob.glob('src_all/sample_data/16*'),
                    stack=True,
                    name='OPT data')
    napari.run()
