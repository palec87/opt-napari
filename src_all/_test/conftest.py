import pytest
import numpy as np
from ..backtrack import Backtrack


@pytest.fixture(scope='function')
def backtrack_true_false():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=True, track_value=False)
    yield backtrack
    del backtrack


@pytest.fixture(scope='function')
def backtrack_none_none():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=None, track_value=None)
    yield backtrack
    del backtrack


@pytest.fixture(scope='function')
def backtrack_true_true():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=True, track_value=True)
    yield backtrack
    del backtrack


@pytest.fixture(scope='function')
def backtrack_false_true():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=False, track_value=True)
    yield backtrack
    del backtrack


@pytest.fixture(scope='function')
def backtrack_false_false():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=False, track_value=False)
    yield backtrack
    del backtrack


@pytest.fixture(scope='function')
def backtrack_1_0():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=1, track_value=0)
    yield backtrack
    del backtrack


# fixture for image layer, dark and bright layer and bad pixel layer
@pytest.fixture(scope='function')
def data1():
    img = np.ones((10, 5, 5)) * 10
    dark = np.ones((5, 5)) * 0.1
    bright = np.ones((5, 5)) * 11
    bad_px = np.ones((5, 5)) * 0.1
    bad_px[2, 1] = 10
    return img, dark, bright, bad_px


# fixture for image layer, dark and bright layer and bad pixel layer
# Not very reasonable but kinda valid
@pytest.fixture(scope='function')
def data2():
    img = np.ones((10, 5, 5)) * 10
    dark = np.ones((5, 5)) * 0.1
    bright = np.ones((5, 5)) * 8.8
    bad_px = np.ones((5, 5)) * 0.1
    bad_px[2, 1] = 10
    return img, dark, bright, bad_px


# @pytest.fixture(scope='module', params=[
#         ({'inplace': True, 'track': False}),
#         ({'inplace': None, 'track': None}),
#         ({'inplace': True, 'track': True}),
#         ({'inplace': False, 'track': True}),
#         ({'inplace': False, 'track': False}),
#         ({'inplace': 1, 'track': 0}),
#     ])
# def backtrack(request):
#     backtrack = Backtrack()
#     backtrack.set_settings(request.param['inplace'],
#                            request.param['track'],
#                            )
#     return backtrack
