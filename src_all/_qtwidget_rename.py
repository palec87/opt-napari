import numpy as np

import napari
from napari.layers import Image, Points
from napari.utils import progress, notifications
from magicgui.widgets import create_widget

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLineEdit,
    QRadioButton, QLabel,
    QButtonGroup, QGroupBox,
    QMessageBox, QDialog, QSizePolicy,
)

from enum import Enum
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)

from widget_settings import Settings, Combo_box
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


class neighbours_choice_modes(Enum):
    n4 = 1
    n8 = 2


# TODO: Correct() as a class attribute? Right now it is created in each method
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
        self.history = Backtrack()
        self.setup_ui()

        # setup_ui initializes track and inplace values
        # here I update history instance flags
        self.updateHistoryFlags()

    def correctDarkBright(self):
        # display message in the message box
        self.messageBox.setText('Correcting Dark and Bright')

        # TODO: inplace and tracking needs to be taken care of here.

        original_image = self.image_layer_select.value
        dark = self.dark_layer_select.value.data
        bright = self.bright_layer_select.value.data
        data_corr = np.zeros(original_image.data.shape,
                            #  dtype=original_image.data.dtype, # I do not like not keeping the dtyp the same, but float operations do not work otherwise
                             )
        if self.flagBright.val:
            if self.flagDark.val and self.flagExp == 'Transmission':
                data_corr = ((original_image.data - dark) /
                             (bright - dark)).astype(np.float32)
                # because it is float between 0-1, needs to be rescaled
                # and casted to uint16 check if the image has element
                # greater than 1 or less than 0
                print('How the division works?', original_image.data.shape,
                      bright.shape, dark.shape)

                if np.amax(data_corr) > 1 or np.amin(data_corr) < 0:
                    self.messageBox.setText(
                        'Dark included, image values out of range. Clipping to 0-1.',
                        )
                    print('Overflows', data_corr.min().compute(),
                          data_corr.max().compute())
                    data_corr = np.clip(data_corr, 0, 1)

                data_corr = (data_corr * 65535).astype(np.uint16)

            elif self.flagExp == 'Emission':
                # this is integer, no casting needed
                data_corr = original_image.data - bright
            else:  # transmission, no dark correction
                data_corr = original_image.data / bright
                # this is float between 0-1, rescaled and cast to uint16
                # make sure that the image is between 0-1 first
                if np.amax(data_corr) > 1 or np.amin(data_corr) < 0:
                    self.messageBox.setText(
                        'No dark, image values out of range. Clipping to 0-1.',
                        )
                    data_corr = np.clip(data_corr, 0, 1)
                data_corr = (data_corr * 65535).astype(np.uint16)

        elif self.flagDark.val:  # only dark correction
            # this is integer, no casting needed
            data_corr = original_image.data - dark
        else:
            self.messageBox.setText(
                'No correction selected.',
                )
            data_corr = original_image.data

        # history update
        if self.history.inplace:
            new_data = {'operation': 'corrDB',
                        'data': data_corr,
                        }
            data_corr = self.history.update_history(original_image,
                                                    new_data)

        self.show_image(data_corr,
                        'Dark-Bright-corr_' + original_image.name,
                        original_image.contrast_limits)

    # hot pixels correction (NOT WORKING YET)
    def correctBadPixels(self):
        """Corrects hot pixels in the image.

        This method corrects hot pixels in the image using a correction
        algorithm. It calculates the number of hot pixels and dead pixels,
        and provides an option to either correct the hot pixels or display
        them. If the correction is performed in-place, it also
        updates the history of the image.

        Returns:
            None
        """
        original_image = self.image_layer_select.value

        # init correction class
        self.messageBox.setText('Correcting hot pixels.')
        corr = Correct(hot=self.hot_layer_select.value.data,
                       std_mult=self.std_cutoff.val)
        hotPxs, deadPxs = corr.get_bad_pxs(mode=self.flagBad)

        self.messageBox.setText(f'Number of hot pixels: {len(hotPxs)}')
        self.messageBox.setText(f'Number of dead pixels: {len(deadPxs)}')

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Bad Pixel correction")
        dlg.setText("Do you want to correct all those pixels! \n"
                    "It can take a while. \n"
                    f"{len(hotPxs)} + {len(deadPxs)}")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        button = dlg.exec()

        # yes/no dialog to ask if the user wants to correct the hot pixels
        if button == QMessageBox.No:
            # display bad pixels
            data_corr = np.zeros(self.hot_layer_select.value.data.shape,
                                 dtype=original_image.data.dtype)
            # hot pixels and dead pixels are displayed as 100
            for _i, (y, x) in enumerate(hotPxs):
                data_corr[y, x] = 100
            for _i, (y, x) in enumerate(deadPxs):
                data_corr[y, x] = 101
            self.show_image(data_corr,
                            'Bad_pixels_' + self.hot_layer_select.value.name,
                            # self.hot_layer_select.value.contrast_limits,
                            )
            return

        # Correction is done, TODO: ooptimized for the volumes and threaded
        data_corr = np.zeros(original_image.data.shape,
                             dtype=original_image.data.dtype)
        for i, img in progress(enumerate(original_image.data)):
            data_corr[i] = corr.correctBadPxs(img)
            print(i)

        # history update
        if self.history.inplace:
            new_data = {'operation': 'corrBP',
                        'data': data_corr,
                        'mode': self.flagBad,
                        'hot_pxs': hotPxs,
                        'dead_pxs': deadPxs,
                        }
            data_corr = self.history.update_history(original_image, new_data)

        self.show_image(data_corr,
                        'Bad-px-corr_' + original_image.name,
                        original_image.contrast_limits)

    def correctIntensity(self):
        """Corrects the intensity of the image.

        This method performs intensity correction on the selected image layer.
        It uses the `Correct` class to perform the correction and updates the
        image data accordingly. If the correction is performed in-place, it
        also updates the history of the image.
        """
        self.messageBox.setText('Correcting intensity.')
        corr = Correct(bright=self.bright_layer_select.value.data)

        # data to correct
        original_image = self.image_layer_select.value
        data_corr = np.zeros(original_image.data.shape,
                             dtype=original_image.data.dtype)

        # intensity correction
        # TODO: use_bright should be a setting, not urgent
        data_corr, corr_dict = corr.correct_int(
                                    original_image.data,
                                    use_bright=False,
                                    rect_dim=self.rectSize.val)

        # TODO: open widget with a intensity plots
        self.plotDialog = PlotDialog(self, corr_dict)
        self.plotDialog.resize(800, 400)
        self.plotDialog.show()

        # history update
        if self.history.inplace:
            new_data = {'operation': 'corrInt',
                        'data': data_corr,
                        'rect_dim': self.rectSize.val,
                        }
            data_corr = self.history.update_history(original_image, new_data)

        self.show_image(data_corr,
                        'Int-corr_' + original_image.name,
                        original_image.contrast_limits)

    def select_ROIs(self):
        """Selects regions of interest (ROIs) based on the given points.

        This method takes the selected image layer and the points layer as
        input. It calculates the upper-left corner coordinates of the last
        point in the points layer, and displays them in a message box. Then,
        it uses the `select_roi` function to extract the selected ROI from
        the original image data, based on the last point, ROI height, and ROI
        width. If the `inplace` flag is set in the `history` object,
        it updates the history with the ROI operation.
        """
        original_image = self.image_layer_select.value
        try:
            points = self.points_layer_select.value.data
        except AttributeError:
            self.messageBox.setText('No points layer selected.')
            return
        self.messageBox.setText(
            f'UL corner coordinates: {points[0][1:].astype(int)}')

        # DP: do I need roi pars for tracking?
        selected_roi, roi_pars = select_roi(original_image.data,
                                            points[-1],  # last point
                                            self.roi_height.val,
                                            self.roi_width.val)

        # history update
        if self.history.inplace:
            new_data = {'operation': 'roi',
                        'data': selected_roi,
                        'roi_def': roi_pars,
                        }
            selected_roi = self.history.update_history(original_image,
                                                       new_data)

        self.show_image(selected_roi,
                        'ROI_' + original_image.name,
                        original_image.contrast_limits)

    def bin_stack(self):
        """Bins the selected image stack by the specified bin factor.

        If the bin factor is 1, nothing is done and an info notification is
        shown. Otherwise, the selected image stack is binned by the bin factor
        and the binned stack is displayed. The original and binned stack shapes
        are also shown in an info notification.

        If the history is set to inplace, the binning operation is added to the
        history with the updated binned stack.
        """
        if self.bin_factor.val == 1:
            notifications.show_info('Bin factor is 1, nothing to do.')
            return

        original_image = self.image_layer_select.value
        binned_roi = bin_3d(original_image.data, self.bin_factor.val)
        notifications.show_info(
            f'Original shape: {original_image.data.shape},'
            f'binned shape: {binned_roi.shape}')

        # history update
        if self.history.inplace:
            new_data = {'operation': 'bin',
                        'data': binned_roi,
                        'bin_factor': self.bin_factor.val,
                        }
            binned_roi = self.history.update_history(original_image, new_data)

        self.show_image(binned_roi,
                        'binned_' + original_image.name,
                        original_image.contrast_limits)

    def calcLog(self):
        """Calculate the logarithm of the image data.

        This method calculates the logarithm of the image data and updates
        the displayed image accordingly. It also updates the history if the
        operation is performed in-place.
        """
        self.messageBox.setText('Calculating -Log')
        original_image = self.image_layer_select.value
        log_image = -np.log10(original_image.data)

        if self.history.inplace:
            new_data = {'operation': 'log',
                        'data': log_image,
                        }
            log_image = self.history.update_history(original_image, new_data)

        self.show_image(log_image,
                        '-Log_' + original_image.name,
                        )

    def undoHistory(self):
        """Undo the last operation in the history.

        This method undoes the last operation in the history and updates the
        displayed image accordingly. If the history is empty, an info
        notification is shown.
        """
        last_op = self.history.history_item['operation']
        self.image_layer_select.value.data = self.history.undo()

        # reset contrast limits
        self.image_layer_select.value.reset_contrast_limits()
        self.messageBox.setText(f'Undoing {last_op}')

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
            if contrast is not None:
                self.viewer.layers[fullname].contrast_limits = contrast
            else:
                self.viewer.layers[fullname].reset_contrast_limits()

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

    def updateHistoryFlags(self):
        self.history.set_settings(self.inplace.val, self.track.val)

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

        # layout for radio buttons
        boxSetting = QHBoxLayout()
        self.inplace = Settings('Inplace operations',
                                dtype=bool,
                                initial=True,
                                layout=boxSetting,
                                write_function=self.set_preprocessing)

        self.track = Settings('Track',
                              dtype=bool,
                              initial=False,
                              layout=boxSetting,
                              write_function=self.set_preprocessing)

        box.addLayout(boxSetting)
        # undo button
        self.undoBtn = QPushButton('Undo')
        self.undoBtn.clicked.connect(self.undoHistory)
        box.addWidget(self.undoBtn)

        # make self.track visible only when self.inplace is True
        slayout.addWidget(groupbox)
        self.setTrackVisibility()
        self.inplace.sbox.stateChanged.connect(self.updateHistoryFlags)
        self.track.sbox.stateChanged.connect(self.updateHistoryFlags)

    def setTrackVisibility(self):
        # initial setting
        self.track.sbox.setVisible(self.inplace.val)
        self.track.lab.setVisible(self.inplace.val)
        self.undoBtn.setVisible(self.track.val)

        # update visibility of track box when inplace box is changed
        self.inplace.sbox.stateChanged.connect(
            lambda: self.track.sbox.setVisible(self.inplace.val))
        # update the label visibility of track box when inplace box is changed
        self.inplace.sbox.stateChanged.connect(
            lambda: self.track.lab.setVisible(self.inplace.val))

        # update visibility of undoBtn when track box is changed
        self.track.sbox.stateChanged.connect(
            lambda: self.undoBtn.setVisible(self.track.val))

    def updateExperimentMode(self, button):
        self.flagExp = button.text()
        self.messageBox.setText(f'Experiment modality: {self.flagExp}')

    def updateBadPixelsMode(self, button):
        self.flagBad = badPxDict[button.text()]
        self.messageBox.setText(f'Bad pixels option: {self.flagBad}')

    def createSettings(self, slayout):
        # message box
        self.messageBox = QLineEdit()
        self.messageBox.setReadOnly(True)
        self.messageBox.setText('Messages')

        # Linear structure to impose correct order
        groupbox1 = QGroupBox('Dark/Bright Correction')
        boxAll = QVBoxLayout()
        boxExp = QHBoxLayout()
        groupbox1.setLayout(boxAll)

        # create QradioButtons
        groupExpMode = QButtonGroup(self)
        groupExpMode.buttonClicked.connect(self.updateExperimentMode)

        transExp = QRadioButton('Transmission')
        transExp.setChecked(True)
        groupExpMode.addButton(transExp)
        boxExp.addWidget(transExp)

        emlExp = QRadioButton('Emmision')
        groupExpMode.addButton(emlExp)
        boxExp.addWidget(emlExp)
        boxAll.addLayout(boxExp)

        # update flag
        self.updateExperimentMode(groupExpMode.checkedButton())

        boxInclude = QHBoxLayout()
        # checkboxes
        self.flagDark = Settings('Include Dark',
                                 dtype=bool,
                                 initial=False,
                                 layout=boxInclude,
                                 write_function=self.set_preprocessing)
        self.flagBright = Settings('Include Bright',
                                   dtype=bool,
                                   initial=False,
                                   layout=boxInclude,
                                   write_function=self.set_preprocessing)
        boxAll.addLayout(boxInclude)
        # Now comes button
        self.addButton(boxAll, 'Correct Dark+Bringht', self.correctDarkBright)
        slayout.addWidget(groupbox1)

        # create a groupbox for hot pixel correction
        groupbox2 = QGroupBox('Bad pixels correction')
        box2 = QVBoxLayout()
        groupbox2.setLayout(box2)

        # Hot pixel correction
        self.neigh_mode = Combo_box(name='Mode', 
                                    choices=neighbours_choice_modes, 
                                    layout=box2,
                                    write_function=self.reset_choices)  # TODO check if reset_choices is correct
        self.std_cutoff = Settings('Hot STD cutoff',
                                   dtype=int,
                                   initial=5,
                                   vmin=1,
                                   vmax=20,
                                   layout=box2,
                                   write_function=self.set_preprocessing)

        groupBadPxMode = QButtonGroup(self)
        groupBadPxMode.buttonClicked.connect(self.updateBadPixelsMode)

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

        self.updateBadPixelsMode(groupBadPxMode.checkedButton())

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

        # TODO: set message bos scrollable or stretchable, this below does not work
        slayout.setSizeConstraint(QVBoxLayout.SetNoConstraint)
        slayout.addWidget(self.messageBox, stretch=3)

    def addButton(self, layout, button_name, call_function):
        button = QPushButton(button_name)
        button.clicked.connect(call_function)
        layout.addWidget(button)


class PlotDialog(QDialog):
    """
    Create a pop-up widget with the OPT time execution
    statistical plots. Timings are collected during the last run
    OPT scan. The plots show the relevant statistics spent
    on particular tasks during the overall experiment,
    as well as per OPT step.
    """
    def __init__(self, parent, report: dict) -> None:
        super(PlotDialog, self).__init__(parent)
        self.mainWidget = QWidget()
        layout = QVBoxLayout(self.mainWidget)
        canvas = IntCorrCanvas(report, self.mainWidget, width=300, height=300)
        layout.addWidget(canvas)
        self.setLayout(layout)


class IntCorrCanvas(FigureCanvas):
    def __init__(self, data_dict, parent=None, width=300, height=300):
        """ Plot of the report

        Args:
            corr_dict (dict): report data dictionary
            parent (_type_, optional): parent class. Defaults to None.
            width (int, optional): width of the plot in pixels.
                Defaults to 300.
            height (int, optional): height of the plot in pixels.
                Defaults to 300.
        """
        fig = Figure(figsize=(width, height))
        self.ax1 = fig.add_subplot()
        # self.ax2 = fig.add_subplot(132)
        # self.ax3 = fig.add_subplot(133)

        self.createFigure(data_dict)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def createFigure(self, data_dict: dict) -> None:
        """Create report plot.

        Args:
            data_dict (dict): Intensity correction dictionary.
        """
        my_cmap = mpl.colormaps.get_cmap("viridis")
        colors = my_cmap(np.linspace(0, 1, 2))
        self.ax1.plot(data_dict['stack_orig_int'],
                      label='Original',
                      color=colors[0])

        self.ax1.plot(data_dict['stack_corr_int'],
                      label='Corrected',
                      color=colors[1])

        self.ax1.set_xlabel('OPT step number')
        self.ax1.set_ylabel('Intensity')
        self.ax1.legend()


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
        # set image layer to OPT data
        optWidget.image_layer_select.value = viewer.layers['OPT data']
        optWidget.hot_layer_select.value = viewer.layers['hot']
        optWidget.dark_layer_select.value = viewer.layers['dark']
        optWidget.bright_layer_select.value = viewer.layers['bright']
    napari.run()
