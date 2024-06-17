#!/usr/bin/env python

"""
Corrections module for the microscopy acquisition

corrections available:
#. Dark-field correction
    * Always applicable and should be always performed
#. Flat-field, ie bright-field correction
    * Important for transmission measurements. Performs badly
        on the floating elements in the FOV
    * Harder to do for the Fluorescence, because of the excitation scatter
        leaking through filters.
    * Flat field can be used to identify dead pixels too
#. Hot pixel correction
    * Hot pixels obtained from long exposure on the blocked camera
    * Dead pixels form flat field acquisition.
    * TODO Notimplemented: Alternative you can use it for dead pixel
        identification (rename to BadPixel correction)
    * Possible to correct as mean of 4 or 8 neighbours
TODO: Intensity correction
More crucial for the shorter exposure times (less averaging of the source
intensity variations), which depends on which time-scale the light-source
drifts.

Notes:
* Exposure on the dark and bright field corrections must be
the same as the experimental exposure.
* TODO: Need to ensure correct logic of the correction and their redefinition.
    Is the logic supposed to be taken care of on this class level?
"""

import numpy as np
from utils import norm_img, img_to_int_type, is_positive


class Correct():
    """Correcting raw data from 2D array acquisitions. Currently implemented
    corrections are:

    #. Dark-field
    #. Bright-field
    #. Hot-pixel correction
    #. Intensity correction.
    """
    def __init__(self, hot: np.array = None, std_mult: float = 7.,
                 dark: np.array = None, bright: np.array = None,
                 ) -> None:
        """ Initialize the correction class with the correction arrays.

        Args:
            hot (np.array, optional): Hot pixel acquisiion. Defaults to None.
            std_mult (float, optional): STD cutoff for outliers (hot pixels).
                Defaults to 7..
            dark (np.array, optional): Dark counts camera acquisition.
                Defaults to None.
            bright (np.array, optional): Bright field correction acquisiiont.
                Defaults to None.
        """
        self.hot = hot
        self.hot_pxs = None
        self.std_mult = std_mult

        self.dark = dark
        self.dark_corr = None

        self.bright = bright
        self.bright_corr = None

        if hot is not None:
            print('Change, get_bad_pixels is a new name',
                  'needs to be called but allows to identify.',
                  'dead pixels for FL hot pixels for TR')

            # self.hot_pxs = self.get_hot_pxs()

    def get_bad_pxs(self, mode: str = 'hot') -> list[tuple[int, int]]:
        """
        Identify hot pixels from the hot array based on the hot
        std_mutl facter threshold. Hot pixel has intensity greater than

        mean(img) + std_mult * std(img)
        Args:
            mode (str, optional): Mode of the hot pixel identification.
                Defaults to 'hot'. Options are 'hot', 'dead' and 'both'.
        """
        if self.hot is None:
            raise ValueError('No hot pixel array provided')

        hot_pxs, dead_pxs = [], []
        self.mean = np.mean(self.hot, dtype=np.float64)
        self.std = np.std(self.hot, dtype=np.float64)
        print('Hot image stats', mode, self.mean, self.std)

        if mode == 'hot' or mode == 'both':
            self.maskAbove = np.ma.masked_greater(
                            self.hot,
                            self.mean + self.std_mult * self.std,
                            )

            # if mask did not get any hot pixels, return empty list
            if np.all(self.maskAbove.mask is False):
                print('No hot pixels identified')
            else:
                # iterate over the mask, and append hot pixels to the list
                for row, col in zip(*np.where(self.maskAbove.mask)):
                    hot_pxs.append((row, col))

        if mode == 'dead' or mode == 'both':
            self.maskBelow = np.ma.masked_less(
                            self.hot,
                            self.mean - self.std_mult * self.std,
                            )

            # if mask did not get any dead pixels, return empty list
            if np.all(self.maskBelow.mask is False):
                print('No dead pixels identified')
            else:
                # iterate over the mask and append dead pixels to the list
                for row, col in zip(*np.where(self.maskBelow.mask)):
                    dead_pxs.append((row, col))
        self.hot_pxs = hot_pxs
        self.dead_pxs = dead_pxs
        return hot_pxs, dead_pxs

    def correctBadPxs(self, img: np.array, mode: str = 'n4') -> np.array:
        """Correct hot pixels from its neighbour pixel values. It ignores the
        neighbour pixel if it was identified as hot pixel itself.

        Args:
            img (np.array): image to be corrected.
            mode (str, optional): How to pick neighbours. Defaults to 'n4',
                up, bottom left, right. Other option is n8, which takes the
                diagonal neighbours too.

        Raises:
            IndexError: Raised if the hot_corr array does not match the shape
                of the input img.
            ValueError: invalid mode option

        Returns:
            np.array: Corrected img array
        """
        self.badPxs = set(self.hot_pxs + self.dead_pxs)

        if self.badPxs == []:
            print('No hot pixels identified, nothing to correct')
            return img

        # check if the shapes of the correction and image match
        if self.hot.shape != img.shape:
            print(self.hot.shape, img.shape)
            raise IndexError('images do not have the same shape')

        # define neighbours
        if mode == 'n4':
            neighs = [(-1, 0), (0, -1), (0, 1), (1, 0)]  # U, L, R, D
        elif mode == 'n8':
            neighs = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1),
                      ]
        else:
            raise ValueError('Unknown mode option, valid is n4 and n8.')

        ans = img.copy()

        # loop over identified hot pixels and correct
        for badPx in self.badPxs:
            neigh_vals = []
            for neigh in neighs:
                px = np.add(np.array(badPx), np.array(neigh))
                # I can do this because I checked shapes above
                # check if neighbour is out of the image ranges.
                if 0 > px[0] or px[0] >= img.shape[0] or 0 > px[1] or px[1] >= img.shape[1]:
                    continue

                # ignore if neighbour is hot pixel
                if tuple(px) in self.hot_pxs:
                    continue

                neigh_vals.append(img[px[0], px[1]])

            # replace hot pixel with the mean of the neighbours
            ans[badPx] = int(np.mean(neigh_vals))

        # test for negative values
        is_positive(ans, 'Bad-pixel')

        # cast it on correct dtype
        ans = img_to_int_type(ans, dtype=ans.dtype)
        return ans

    def correct_dark(self, img: np.array) -> np.array:
        """Subtract dark image from the img.
        TODO: treating if dark correction goes negative?? Ask if to continue?

        Args:
            img (np.array): Img to be corrected

        Raises:
            IndexError: If the shapes do not match

        Returns:
            np.array: corrected image
        """
        if self.dark.shape != img.shape:
            raise IndexError('images do not have the same shape')

        # correction
        ans = img - self.dark
        # test for negative values
        is_positive(ans, 'Dark-field')

        # cast it on correct dtype
        ans = img_to_int_type(ans,  dtype=img.dtype)
        return ans

    def correct_bright(self, img: np.array) -> np.array:
        """Correct image using a bright-field correction image

        Args:
            img (np.arrays): Img to correct

        Raises:
            IndexError: If the shapes do not match

        Returns:
            np.array: Corrected image
        """
        if self.bright.shape != img.shape:
            raise IndexError('images do not have the same shape')

        # bright-field needs to be first corrected with
        # dark and hot pixels if possible
        try:
            self.bright_corr = norm_img(self.bright_corr)
        except:
            print('Probably bright is not yet dark/hot corrected, trying that')
            # TODO: ensure this is done only once. Should offer redoing it
            # from the raw image, if user tries to run this second time.
            self.bright_corr = self.correct_dark(self.bright)      
            if self.hot is None:
                pass
            else:
                try:
                    # the same as above, run only once.
                    self.bright_corr = self.correctBadPxs(self.bright_corr)
                except TypeError:
                    pass

            # normalize to one (return floats)
            self.bright_corr = norm_img(self.bright_corr)
        # not overflow, because a float
        ans = img / self.bright_corr

        # test for negative values
        is_positive(ans, 'Bright-field')

        # cast it on correct dtype, and clips negative values!!
        ans = img_to_int_type(ans, dtype=img.dtype)
        return ans

    def correct_int(self, img_stack: np.ndarray, mode: str = 'integral',
                    use_bright: bool = True, rect_dim: int = 50,
                    cast_to_int: bool = True) -> np.ndarray:
        """Intensity correction over the stack of images which are expected
        to have the same background intensity. It is preferable to
        corrected for dark, bright, and bad pixels first

        Args:
            img_stack (np.array): 3D array of images, third dimension is
                along angles
            mode (str, optional): correction mode, only available is integral
                and integral_bottom. Defaults to 'integral'.
            use_bright (bool, optional): if bright field acquisition is a ref
                to scale images. Defaults to True.
            rect_dim (int, optional): size of rectabgles in the corners in
                pixels. Defaults to 50.

        Raises:
            NotImplementedError: Checking available correction modes

        Returns:
            np.ndarray: 3D array of the same shape as the img_stack,
                but intensity corrected
        """
        # check if stack is 3D array
        if img_stack.ndim != 3:
            raise IndexError('Stack has to have three dimensions.')
        # do I want to correct in respect to the bright field
        # basic idea is four corners, integrated
        # second idea, fit a correction plane into the four corners.
        if use_bright is True and self.bright is not None:
            # four corners of the bright
            # this is useless option!!!
            ref = ((self.bright[:rect_dim, :rect_dim]),
                   (self.bright[:rect_dim, -rect_dim:]),
                   (self.bright[-rect_dim:, :rect_dim]),
                   (self.bright[-rect_dim:, -rect_dim:]),
                   )
        else:
            print('Using avg of the corners in the img stack as ref')
            # assuming the stacks 3rd dimension is the right one.
            # mean over steps in the aquisition
            ref = ((np.mean(img_stack[:, :rect_dim, :rect_dim], axis=0)),
                   (np.mean(img_stack[:, :rect_dim, -rect_dim:], axis=0)),
                   (np.mean(img_stack[:, -rect_dim:, :rect_dim], axis=0)),
                   (np.mean(img_stack[:, -rect_dim:, -rect_dim:], axis=0)),
                   )

        print('shape ref:', [k.shape for k in ref])

        # integral takes sum over pixels of interest
        # TODO: This if else structure is cumbersome
        if mode == 'integral':
            # sum of all pixels over all four squares
            # this is one number
            self.ref = np.mean([np.mean(k) for k in ref])
        elif mode == 'integral_bottom':
            self.ref = np.mean([np.mean(ref[2]), np.mean(ref[3])])
        else:
            raise NotImplementedError

        # correct the stack
        corr_stack = np.empty(img_stack.shape, dtype=img_stack.dtype)

        # intensity numbers for img in the stack (sum over ROIs)
        intOrig, intCorr = [], []
        for i, img in enumerate(img_stack):
            # two means are not a clean solution
            # as long as the rectangles ar the same, it is equivalent
            if mode == 'integral':
                img_int = np.mean((np.mean(img[:rect_dim, :rect_dim]),
                                np.mean(img[:rect_dim, -rect_dim:]),
                                np.mean(img[-rect_dim:, :rect_dim]),
                                np.mean(img[-rect_dim:, -rect_dim:]),
                                ))
            elif mode == 'integral_bottom':
                img_int = np.mean((
                                np.mean(img[-rect_dim:, :rect_dim]),
                                np.mean(img[-rect_dim:, -rect_dim:]),
                ))
            intOrig.append(img_int)
            corr_stack[i] = (img / img_int) * self.ref

            # intensity after correction
            intCorr.append(np.mean((np.mean(corr_stack[i][:rect_dim, :rect_dim]),
                                    np.mean(corr_stack[i][:rect_dim, -rect_dim:]),
                                    np.mean(corr_stack[i][-rect_dim:, :rect_dim]),
                                    np.mean(corr_stack[i][-rect_dim:, -rect_dim:]),
                                    ))
                           )
            print(i, end='\r')

        # stored in order to tract the stability fluctuations.
        intOrig = np.array(intOrig)
        intCorr = np.array(intCorr)

        # test for negative values
        is_positive(corr_stack, 'Intensity')

        # cast it on correct dtype
        if cast_to_int:
            corr_stack = img_to_int_type(corr_stack, dtype=corr_stack.dtype)

        # int correction dictioanary report
        intCorrReport = {'mode': mode,
                         'use_bright': use_bright,
                         'rect_dim': rect_dim,
                         'ref': self.ref,
                         'stack_orig_int': intOrig,
                         'stack_corr_int': intCorr,
                         }

        return corr_stack, intCorrReport

    # This method not used in the napari plugin
    def correct_all(self, img: np.array, mode_hot='n4') -> np.array:
        """
        Perform all available corrections for single np.array image.

        1. subtract dark from img and white
        2. remove hot pixels from white
        3. correct white in the img
        4. remove hot pixels from img

        Args:
            img (np.array): img to correct
            mode_hot (str, optional): mode of selecting hot pixel neighbours.
                Defaults to 'n4'.

        Returns:
            np.array: Fully corrected image, also casted to the original dtype
        """
        # TODO: this leads to repetition of steps I think
        img_format = img.dtype

        # step 1
        self.img_corr = self.correct_dark(img)
        self.bright_corr = self.correct_dark(self.bright)  # this could be done only once

        # step 2
        self.bright_corr = self.correctBadPxs(self.bright_corr, mode=mode_hot)  # this too
        self.bright_corr = norm_img(self.bright_corr)

        # step 3
        self.img_corr = self.correct_bright(self.img_corr)

        # step 4
        self.img_corr = self.correctBadPxs(self.img_corr)

        # step 5, make sure it is the same integer type as original img.
        self.img_corr = img_to_int_type(self.img_corr, img_format)

        return self.img_corr
