#!/usr/bin/env python

"""
Tests for correction class
"""

import pytest
import numpy as np
from ..corrections import Correct


__author__ = 'David Palecek'
__credits__ = ['Teresa M Correia', 'Giorgia Tortora']
__license__ = 'GPL'

hot1 = np.ones((5, 5))


@pytest.mark.parametrize(
    'init_vals, expected',
    [((), (None, 7, None, None)),
     ((None, ), (None, 7, None, None)),
     ((hot1, ), ([], 7, None, None)),
     ],
)
def test_init(init_vals, expected):
    corr = Correct(*init_vals)
    assert corr.hot_pxs == expected[0]
    assert corr.std_mult == expected[1]
    assert corr.dark is expected[2]
    assert corr.bright_corr is expected[3]
