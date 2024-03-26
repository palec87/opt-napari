#!/usr/bin/env python

"""
Testing the low level helper functions
"""

import pytest
import numpy as np
from ..utils import (
    select_roi, bin_3d, norm_img,
    img_to_int_type, is_positive)


__author__ = 'David Palecek'
__credits__ = ['Teresa M Correia', 'Giorgia Tortora']
__license__ = 'GPL'


# TODO: more test cases
@pytest.mark.parametrize(
    'arr_in, ULCorner, height, width, arr_out, roi_out',
    [(np.ones((5, 5)), (2, 2), 2, 2, np.ones((2, 2)), (2, 4, 2, 4)),
     ],
)
def test_select_roi(arr_in, ULCorner, height, width, arr_out, roi_out):
    out, roi_pars = select_roi(arr_in, ULCorner, height, width)
    assert roi_pars == roi_out
    np.testing.assert_array_equal(out, arr_out)


@pytest.mark.parametrize(
    'arr_in, bin_factor, expected',
    [(np.ones((5, 4, 4)), 2, np.ones((5, 2, 2))),
     ],
    )
def test_bin_3d(arr_in, bin_factor, expected):
    out = bin_3d(arr_in, bin_factor)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    'arr_in, ret_type, expected',
    [(2 * np.ones((4, 4)), 'float', np.ones((4, 4))),
     ],
    )
def test_norm_img(arr_in, ret_type, expected):
    out = norm_img(arr_in, ret_type)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    'arr_in, dtype, expected',
    [(np.ones((4, 4)), np.uint8, np.ones((4, 4))),
     ],
    )
def test_img2intType(arr_in, dtype, expected):
    out = img_to_int_type(arr_in, dtype)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    'arr_in, corr_type, expected',
    [(np.ones((4, 4)), 'dark_corr', 0),
     ],
    )
def testIsPositive(arr_in, corr_type, expected):
    out = is_positive(arr_in, corr_type)
    assert out == expected
