#!/usr/bin/env python

"""
Tests for backtrack class
"""

import pytest
from napari import Viewer
import numpy as np
from ..backtrack import Backtrack


__author__ = 'David Palecek'
__credits__ = ['Teresa M Correia', 'Giorgia Tortora']
__license__ = 'GPL'


@pytest.mark.parametrize(
    'input_vals, expected',
    [((True, False), (True, False, True, False)),
     ((None, None), (True, False, True, False)),
     ((True, True), (True, False, True, True)),
     ((False, True), (True, False, False, True)),
     ((False, False), (True, False, False, False)),
     ((1, 0), (True, False, True, False)),
     ],
)
def test_set_settings(input_vals, expected):
    backtrack = Backtrack()

    # this is default, always the same
    assert backtrack.inplace == expected[0]
    assert backtrack.track == expected[1]

    backtrack.set_settings(*input_vals)

    assert backtrack.inplace == expected[2]
    assert backtrack.track == expected[3]


# this works
@pytest.mark.parametrize(
    'input_vals, expected',
    [((True, False), False),
     ((None, None), False),
     ((True, True), True),
     ((False, True), False),
     ((False, False), False),
     ((1, 0), False),
     ],
)
def test_update_compatible(input_vals, expected):
    backtrack = Backtrack()
    backtrack.set_settings(*input_vals)

    assert backtrack._update_compatible() is expected


# TODO: use fixture in order to avoid the backtrack hanging
@pytest.mark.parametrize(
    'input_vals, expected',
    [((True, False, np.ones((5, 5))), (False, {})),
     ((None, None, np.ones((5, 5))), (False, {})),
     ((True, True, np.ones((5, 5))), (True, {'operation': 'roi', 'data': np.ones((5, 5)), 'roi_def': ()})),
     ((False, True, np.ones((5, 5))), (False, np.ones((2, 2)))),
     ((False, False, np.ones((5, 5))), (False, np.ones((2, 2)))),
     ((1, 0, np.ones((5, 5))), (False, np.ones((2, 2)))),
     ],
)
class TestClass:
    def test_update_history_roi(self, make_napari_viewer, input_vals, expected):
        backtrack = Backtrack()
        backtrack.set_settings(input_vals[0], input_vals[1])
        assert backtrack.raw_data is None
        assert backtrack.history_item == {}
        assert backtrack._update_compatible() is expected[0]

        img_old = input_vals[2]
        img_new = np.ones((2, 2))
        viewer: Viewer = make_napari_viewer()
        viewer.add_image(img_old, name='test')
        data_dict = {'operation': 'roi',
                     'data': img_new,
                     'roi_def': (1, 1, 3, 3),
                     }
        out = backtrack.update_history(viewer.layers['test'],
                                       data_dict)


# expected is _update_compatible bool value, dict of {}
@pytest.mark.parametrize(
    'input_vals, expected',
    [((True, False, np.ones((5, 5))), (False, {})),
     ((None, None, np.ones((5, 5))), (False, {})),
     ((True, True, np.ones((5, 5))), (True, {'operation': 'roi', 'data': np.ones((5, 5)), 'roi_def': ()})),
     ((False, True, np.ones((5, 5))), (False, np.ones((2, 2)))),
     ((False, False, np.ones((5, 5))), (False, np.ones((2, 2)))),
     ((1, 0, np.ones((5, 5))), (False, np.ones((2, 2)))),
     ],
)
def test_update_history_roi(make_napari_viewer, input_vals, expected):
    backtrack = Backtrack()
    backtrack.set_settings(input_vals[0], input_vals[1])
    assert backtrack.raw_data is None
    assert backtrack.history_item == {}
    assert backtrack._update_compatible() is expected[0]

    img_old = input_vals[2]
    img_new = np.ones((2, 2))
    viewer: Viewer = make_napari_viewer()
    viewer.add_image(img_old, name='test')
    data_dict = {'operation': 'roi',
                 'data': img_new,
                 'roi_def': (1, 1, 3, 3),
                 }

    out = backtrack.update_history(viewer.layers['test'],
                                   data_dict)

    # assert np.array_equal(out, data_dict['data'])
    # assert np.array_equal(backtrack.raw_data, viewer.layers['test'].data)
    # np.testing.assert_equal(backtrack.history_item, expected[1])
    # assert backtrack.history_item == expected
    del backtrack
