#!/usr/bin/env python

"""
Testing the low level helper functions
TODO: fix binning based on tests
TODO: add test cases, especially edge cases
TODO: functional tests of all corrections in combinations, closer to real life.
"""

import pytest
import numpy as np
from ..utils import (
    select_roi, bin_3d, norm_img,
    img_to_int_type, is_positive)


__author__ = 'David Palecek'
__credits__ = ['Teresa M Correia', 'Giorgia Tortora']
__license__ = 'GPL'


@pytest.mark.parametrize(
    'arr_in, ULCorner, height, width, arr_out, roi_out',
    [(np.ones((5, 5)), (2, 2), 2, 2, np.ones((2, 2)), (2, 4, 2, 4)),
     (np.ones((5, 5)), (2, 2), 3, 3, np.ones((3, 3)), (2, 5, 2, 5)),
     (np.ones((5, 5)), (2, 2), 4, 4, np.ones((3, 3)), (2, 5, 2, 5)),  # ROI beyond the image
     (np.ones((5, 5)), (2, 2), 0, 0, np.ones((0, 0)), (2, 2, 2, 2)),  # empty ROI
     ],
)
def test_select_roi(arr_in, ULCorner, height, width, arr_out, roi_out):
    out, roi_pars = select_roi(arr_in, ULCorner, height, width)
    assert roi_pars == roi_out
    np.testing.assert_array_equal(out, arr_out)


@pytest.mark.parametrize(
    'arr_in, ULCorner, height, width, out',
    [(np.ones((5, 5)), (2, 2), -1, -1, None),  # this should throw an error
     ],
)
def test_select_roi2(arr_in, ULCorner, height, width, out):
    assert select_roi(arr_in, ULCorner, height, width) == out


# this is failing for unmatched dimensions,
@pytest.mark.parametrize(
    'arr_in, bin_factor, expected',
    [(np.ones((5, 4, 4)), 2, np.ones((5, 2, 2))),
     (np.ones((5, 4, 4)), 3, np.ones((5, 1, 1))),
     (np.ones((5, 5, 5)), 3, np.ones((5, 1, 1))),
     (np.ones((5, 6, 6)), 3, np.ones((5, 2, 2))),
     (np.ones((5, 6, 6)), 2, np.ones((5, 3, 3))),
     ],
    )
def test_bin_3d(arr_in, bin_factor, expected):
    out = bin_3d(arr_in, bin_factor)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    'arr_in, ret_type, expected',
    [(2 * np.ones((4, 4)), 'float', np.ones((4, 4))),
     (2 * np.ones((4, 4)), 'int', np.ones((4, 4))),
     (2.5 * np.ones((4, 4)), 'float', np.ones((4, 4))),
     ],
    )
def test_norm_img(arr_in, ret_type, expected):
    out = norm_img(arr_in, ret_type)
    assert out.dtype == ret_type
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    'arr_in, dtype, expected',
    [(np.ones((4, 4)), np.uint8, np.ones((4, 4))),
     (np.ones((4, 4)) * 255, np.uint8, np.ones((4, 4)) * 255),
     (np.ones((4, 4)) * 256, np.uint8, np.ones((4, 4)) * 255),
     ],
    )
def test_img2intType(arr_in, dtype, expected):
    out = img_to_int_type(arr_in, dtype)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    'inputs, expected',
    [((np.ones((4, 4)), 'dark_corr'), 0),
     ((np.ones((4, 4)) - 2, 'dark_corr'), 1),
     ((np.ones((4, 4)) - 2, ), 1),
     ],
    )
def testIsPositive(inputs, expected):
    out = is_positive(*inputs)
    assert out == expected
