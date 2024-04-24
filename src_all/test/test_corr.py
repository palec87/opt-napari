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

hot1 = dark1 = bright1 = np.ones((5, 5))
# define hot one as a copy of hot1 with one pixel changed
hot2 = hot1.copy()
hot2[2, 1] = 10

# define a test image
img1 = np.ones((5, 5)) * 5
img2 = img1.copy()
img2[2, 1] = 1

img3 = np.ones((5, 5)) * 3
img4 = np.ones((5, 5)) * 4

stack1 = np.ones((5, 5, 5)) * 5


@pytest.mark.parametrize(
    'init_vals, expected',
    [((), (None, 7, None, None)),  # no xorrections
     ((None, ), (None, 7, None, None)),
     ((hot1, ), ([], 7, None, None)),
     ((hot2, 5, ), ([], 5, None, None)),  # with five the cutoff is 10.17
     ((hot2, 4.85, ), ([(2, 1)], 4.85, None, None)),
     ],
)
def test_init(init_vals, expected):
    corr = Correct(*init_vals)
    assert corr.hot_pxs == expected[0]
    assert corr.std_mult == expected[1]
    assert corr.dark is expected[2]
    assert corr.bright_corr is expected[3]


@pytest.mark.parametrize(
    'inits, img, expected',
    [((hot1, ), img1, img1),
     ((hot2, 5, ), img1, img1),
     ((hot2, 4, ), img1, img1),  # this does not do anything, because img is flat
     ((hot2, 4, ), img2, img1),  # this is correctimg stuff
     ],
)
def test_correct_hot(inits, img, expected):
    corr = Correct(*inits)
    corrected = corr.correct_hot(img)
    assert corrected.all() == expected.all()


@pytest.mark.parametrize(
    'inits, img, expected',
    [((hot2, 4, dark1), hot1, np.zeros((5, 5))),  # two same arrays
     ((hot2, 4, dark1), img2, img2 - dark1),
     ],
)
def test_correct_dark(inits, img, expected):
    corr = Correct(*inits)
    corrected = corr.correct_dark(img)
    assert corrected.all() == expected.all()


@pytest.mark.parametrize(
    'inits, img, expected',
    [((hot2, 4, dark1, bright1), img1, img4),
     ],
)
def test_correct_bright(inits, img, expected):
    corr = Correct(*inits)
    corrected = corr.correct_bright(img)
    assert corrected.all() == expected.all()

# TODO: check this, I think this should not pass
@pytest.mark.parametrize(
    'inits, img, expected',
    [((hot2, 4, dark1, bright1), img1, img3),  # dark =1, bright = 1, img1 = 5
     ],
)
def test_correct_bright2(inits, img, expected):
    corr = Correct(*inits)
    corrected = corr.correct_dark(img)
    corrected = corr.correct_bright(corrected)
    assert corrected.all() == expected.all()


@pytest.mark.parametrize(
    'inits, corr_int_inputs, expected',
    [((hot2, 4, dark1, bright1), (stack1,), img1),
     ],
)
def test_correct_int(inits, corr_int_inputs, expected):
    corr = Correct(*inits)
    corrected = corr.correct_int(*corr_int_inputs)
    assert corrected.all() == expected.all()
