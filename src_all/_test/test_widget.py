#!/usr/bin/env python

"""
Tests for backtrack class
"""

import pytest
import numpy as np


__author__ = 'David Palecek'
__credits__ = ['Teresa M Correia', 'Giorgia Tortora']
__license__ = 'GPL'


@pytest.mark.parametrize(
    'input_vals, expected',
    [({'flagBright': True, 'flagDark': True, 'flagExp': 'Transmission'},
      ((np.ones((10, 5, 5)) * (10-0.1)/(11-0.1)).clip(0, 1) * 65535).astype(np.uint16)),
     ({'flagBright': False, 'flagDark': True, 'flagExp': 'Transmission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': False, 'flagDark': False, 'flagExp': 'Transmission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': True, 'flagDark': False, 'flagExp': 'Transmission'},
      ((np.ones((10, 5, 5)) * 10 / 11).clip(0, 1) * 65535).astype(np.uint16)), 
     ({'flagBright': True, 'flagDark': True, 'flagExp': 'Emission'},
      (np.zeros((10, 5, 5))).astype(np.uint16)),
     # this should subtract dark only
     ({'flagBright': False, 'flagDark': True, 'flagExp': 'Emission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': False, 'flagDark': False, 'flagExp': 'Emission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': True, 'flagDark': False, 'flagExp': 'Emission'},
      (np.zeros((10, 5, 5))).astype(np.uint16)),
     ],
)
def test_corrDB1(input_vals, expected, request):
    viewer, widget = request.getfixturevalue("prepare_widget_data1")
    widget.flagBright.val = input_vals['flagBright']
    widget.flagDark.val = input_vals['flagDark']
    widget.flagExp = input_vals['flagExp']

    widget.inplace.val, widget.track.val = False, False
    widget.updateHistoryFlags()

    widget.correctDarkBright()
    # assert shape
    assert viewer.layers['img'].data.shape == expected.shape
    # assert dtype
    assert viewer.layers['Dark-Bright-corr_img'].data.dtype == expected.dtype
    # assert data
    assert np.allclose(viewer.layers['Dark-Bright-corr_img'].data, expected)


@pytest.mark.parametrize(
    'input_vals, expected',
    [({'flagBright': True, 'flagDark': True, 'flagExp': 'Transmission'},
      ((np.ones((10, 5, 5)) * (10-0.1)/(8.8-0.1)).clip(0, 1) * 65535).astype(np.uint16)),
     # not sure if this should be 9 or 10
     ({'flagBright': False, 'flagDark': True, 'flagExp': 'Transmission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': False, 'flagDark': False, 'flagExp': 'Transmission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': True, 'flagDark': False, 'flagExp': 'Transmission'},
      ((np.ones((10, 5, 5)) * 10 / 8.8).clip(0, 1) * 65535).astype(np.uint16)),
     ({'flagBright': True, 'flagDark': True, 'flagExp': 'Emission'},
      (np.ones((10, 5, 5))).astype(np.uint16)),
     # this should subtract dark only, should be 9 or 10?
     ({'flagBright': False, 'flagDark': True, 'flagExp': 'Emission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': False, 'flagDark': False, 'flagExp': 'Emission'},
      (np.ones((10, 5, 5)) * 10).astype(np.uint16)),
     ({'flagBright': True, 'flagDark': False, 'flagExp': 'Emission'},
      (np.ones((10, 5, 5))).astype(np.uint16)),
     ],
)
def test_corrDB2(input_vals, expected, request):
    viewer, widget = request.getfixturevalue("prepare_widget_data2")
    widget.flagBright.val = input_vals['flagBright']
    widget.flagDark.val = input_vals['flagDark']
    widget.flagExp = input_vals['flagExp']

    widget.inplace.val, widget.track.val = False, False
    widget.updateHistoryFlags()

    widget.correctDarkBright()
    # assert shape
    assert viewer.layers['img'].data.shape == expected.shape
    # assert dtype
    assert viewer.layers['Dark-Bright-corr_img'].data.dtype == expected.dtype
    # assert data
    assert np.allclose(viewer.layers['Dark-Bright-corr_img'].data, expected)


def test_show_image(request):
    _, widget = request.getfixturevalue("prepare_widget_data1")
    widget.inplace.val, widget.track.val = False, False
    widget.updateHistoryFlags()

    assert widget.history._update_compatible() is False

    widget.show_image(np.ones((5, 5, 5)), 'new_image')
    assert widget.viewer.layers['new_image'].data.shape == (5, 5, 5)
    assert np.array_equal(widget.viewer.layers['new_image'].data,
                          np.ones((5, 5, 5)))


# test clip_and_convert_data method
@pytest.mark.parametrize(
    'input_vals, expected',
    [({'data': np.ones((10, 5, 5)) * 10},
      (np.ones((10, 5, 5)) * 65535).astype(np.uint16),),
     ({'data': np.ones((10, 5, 5)) * -1},
      np.zeros((10, 5, 5)),),
     ({'data': np.array(([[0.1, 5], [0.1, -1]]))},
      (np.array([[0.1, 1], [0.1, 0]]) * 65535).astype(np.uint16),),
     ],
)
def test_clip_and_convert(input_vals, expected, request):
    _, widget = request.getfixturevalue("prepare_widget_data1")
    ans = widget.clip_and_convert_data(input_vals['data'])
    assert np.allclose(ans, expected)
