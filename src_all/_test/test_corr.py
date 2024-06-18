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

hot3 = np.ones((5, 5)) * 10
hot3[1, 2] = 1
hot3[2, 1] = 30

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
     ((hot1, ), (None, 7, None, None)),
     ((hot2, 5, ), (None, 5, None, None)),  # with five the cutoff is 10.17
     ((hot2, 4.85, ), (None, 4.85, None, None)),
     ],
)
def test_init(init_vals, expected):
    corr = Correct(*init_vals)
    assert corr.hot_pxs == expected[0]
    assert corr.std_mult == expected[1]
    assert corr.dark is expected[2]
    assert corr.bright_corr is expected[3]


@pytest.mark.parametrize(
    'init_vals, expected',
    [((), (None, 7, None, None)),  # no xorrections
     ((None, ), (None, 7, None, None)),
     ((hot1, ), ([], 7, None, None)),
     ((hot2, 5, ), ([], 5, None, None)),  # with five the cutoff is 10.17
     ((hot2, 4.85, ), ([(2, 1)], 4.85, None, None)),
     ],
)
def test_get_bad_pxs(init_vals, expected):
    corr = Correct(*init_vals)
    if expected[0] is not None:
        corr.get_bad_pxs()
        assert corr.hot_pxs == expected[0]
        assert corr.std_mult == expected[1]
        assert corr.dark is expected[2]
        assert corr.bright_corr is expected[3]
    else:
        with pytest.raises(ValueError, match='No hot pixel array provided'):
            corr.get_bad_pxs()


@pytest.mark.parametrize(
    'init_vals, mode, expected',
    [((), 'hot', (None, 7, None, None)),  # no corrections
     ((), 'dead', (None, 7, None, None)),  # no corrections
     ((), 'both', (None, 7, None, None)),  # no corrections
     ((), '', (None, 7, None, None)),  # no corrections
     ((None, ), 'hot', (None, 7, None, None)),
     ((None, ), 'dead', (None, 7, None, None)),
     ((None, ), 'both', (None, 7, None, None)),
     ((hot1, ), 'hot', ([], [], 7, None, None)),
     ((hot1, ), 'dead', ([], [], 7, None, None)),
     ((hot1, ), 'both', ([], [], 7, None, None)),
     ((hot1, ), '', ([], [], 7, None, None)),
     ((hot2, 5, ), 'hot', ([], [], 5, None, None)),  # with five the cutoff is 10.17
     ((hot2, 5, ), 'dead', ([], [], 5, None, None)),  # with five the cutoff is 10.17
     ((hot2, 5, ), 'both', ([], [], 5, None, None)),  # with five the cutoff is 10.17
     ((hot2, 5, ), '', ([], [], 5, None, None)),  # with five the cutoff is 10.17
     ((hot2, 4.85, ), 'hot', ([(2, 1)], [], 4.85, None, None)),
     ((hot2, 4.85, ), 'dead', ([], [], 4.85, None, None)),
     ((hot2, 4.85, ), 'both', ([(2, 1)], [], 4.85, None, None)),
     ((hot2, 4.85, ), '', ([(2, 1)], [], 4.85, None, None)),
     ((hot3, 4, ), 'hot', ([(2, 1)], [], 4, None, None)),
     ((hot3, 2, ), 'dead', ([], [(1, 2)], 2, None, None)),
     ((hot3, 2, ), 'both', ([(2, 1)], [(1, 2)], 2, None, None)),
     ((hot3, 3, ), 'both', ([(2, 1)], [], 3, None, None)),
     ((hot3, 4.85, ), '', ([], [], 4.85, None, None)),
     ],
)
def test_get_bad_pxs2(init_vals, mode, expected):
    corr = Correct(*init_vals)
    if expected[0] is None:
        with pytest.raises(ValueError, match='No hot pixel array provided'):
            corr.get_bad_pxs(mode=mode)
    elif mode not in ['hot', 'dead', 'both']:
        with pytest.raises(
                ValueError,
                match='Unknown mode option, valid is hot, dead and both.'):
            corr.get_bad_pxs(mode=mode)
    elif expected[0] is not None:
        corr.get_bad_pxs(mode=mode)
        assert corr.hot_pxs == expected[0]
        assert corr.dead_pxs == expected[1]
        assert corr.std_mult == expected[2]
        assert corr.dark is expected[3]
        assert corr.bright_corr is expected[4]
    else:
        assert 1 == 0


@pytest.mark.parametrize(
    'inits, img, expected',
    [((hot1, ), img1, img1),
     ((hot2, 5, ), img1, img1),
     ((hot2, 4, ), img1, img1),  # no action, because img is flat
     ((hot2, 4, ), img2, img1),  # this is correctimg stuff
     ],
)
def test_correct_hot(inits, img, expected):
    corr = Correct(*inits)
    corr.get_bad_pxs()
    corrected = corr.correctBadPxs(img)
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
    corr.get_bad_pxs()
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
    corr.get_bad_pxs()
    corrected = corr.correct_dark(img)
    corrected = corr.correct_bright(corrected)
    assert corrected.all() == expected.all()


@pytest.mark.parametrize(
    'inits, corr_int_inputs, expected',
    [((hot2, 4, dark1, bright1), (stack1,), (img1, {'mode': 'integral'})),
     ],
)
def test_correct_int(inits, corr_int_inputs, expected):
    corr = Correct(*inits)
    corrected, report = corr.correct_int(*corr_int_inputs)
    assert corrected.all() == expected[0].all()
    assert report['mode'] == expected[1]['mode']
