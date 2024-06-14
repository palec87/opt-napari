#!/usr/bin/env python

"""
Tests for Bad pixel correction because it contains
a dialog window
"""

import pytest
import numpy as np


######################
# Correct Bad Pixels #
######################
@pytest.mark.parametrize(
    'input_vals, expected',
    [({'flagBad': 'hot', 'std': 1}, np.ones((10, 5, 5)) * 10),
     ({'flagBad': 'dead', 'std': 1}, np.ones((10, 5, 5)) * 10),
     ({'flagBad': 'both', 'std': 1}, np.ones((10, 5, 5)) * 10),
     ],
)
def test_bad1(input_vals, expected, request):
    _, widget = request.getfixturevalue("prepare_widget_data1")

    widget.flagBad = input_vals['flagBad']
    widget.std_cutoff.val = input_vals['std']

    widget.inplace.val, widget.track.val = False, False
    widget.updateHistoryFlags()

    # run correction (What about dialog?)
    widget.correctBadPixels()


# with qtbot does not work at all
# @pytest.mark.parametrize(
#     'input_vals, expected',
#     [({'flagBad': 'hot', 'std': 1}, np.ones((10, 5, 5)) * 10),
#      ({'flagBad': 'dead', 'std': 1}, np.ones((10, 5, 5)) * 10),
#      ({'flagBad': 'both', 'std': 1}, np.ones((10, 5, 5)) * 10),
#      ],
# )
# def test_bad1_qtbot(input_vals, expected, request, qtbot):
#     from qtpy import QtWidgets, QtCore, QtTest
#     _, widget = request.getfixturevalue("prepare_widget_data1")
#     qtbot.addWidget(widget)

#     def handle_dialog():
#         # messagebox = QtWidgets.QApplication.activeWindow()
#         # messagebox = widget.activeWindow()
#         # or
#         messagebox = widget.findChild(QtWidgets.QMessageBox)
#         no_button = messagebox.button(QtWidgets.QMessageBox.No)
#         qtbot.mouseClick(no_button, QtCore.Qt.LeftButton, delay=1)

#     widget.flagBad = input_vals['flagBad']
#     widget.std_cutoff.val = input_vals['std']

#     widget.inplace.val, widget.track.val = False, False
#     widget.updateHistoryFlags()

    # # run correction (What about dialog?)
    # widget.correctBadPixels()
    # QtTest.QTest.qWait(500)
    # QtCore.QTimer.singleShot(100, handle_dialog)
    # # qtbot.mouseClick(widget.button, QtCore.Qt.LeftButton, delay=1)
    # QtTest.QTest.qWait(500)
