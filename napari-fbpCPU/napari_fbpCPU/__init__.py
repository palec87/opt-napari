#!/usr/bin/env python


''' Widget for FBP with CPU, threading is an option'''


import numpy as np
from napari.layers import Image
from napari import Viewer
from tqdm import tqdm
from time import perf_counter
from sklearn.linear_model import LinearRegression
import threading

from tomopy.recon.rotation import find_center_vo
import tomopy as tom


def calc_thetas(steps):
    return np.linspace(0., 360., steps, endpoint=False) / 360. * (2 * np.pi)


def run_fbp(data, thetas, center=None):
    height = data.shape[1]
    try:
        r1 = tom.recon(data[:, height//2:height//2+1, :], thetas,
                       center=center[height//2],
                       sinogram_order=False,
                       algorithm='art')
    except IndexError:
        r1 = tom.recon(data[:, height//2:height//2+1, :], thetas,
                       center=center,
                       sinogram_order=False,
                       algorithm='art')
    data_recon = np.zeros((data.shape[1], *r1.squeeze().shape), dtype=np.int16)
    for i in tqdm(range(height)):
        try:
            data_recon[i] = tom.recon(
                                data[:, i:i+1, :], thetas, center=center[i],
                                sinogram_order=False,
                                algorithm='art',
                                ).astype(np.int16)
        except IndexError:
            data_recon[i] = tom.recon(
                                data[:, i:i+1, :], thetas, center=center,
                                sinogram_order=False,
                                algorithm='art',
                                ).astype(np.int16)
    return data_recon


def run_fbp_thread(data, thetas, centers):
    height = data.shape[1]
    r1 = tom.recon(data[:, height//2:height//2+1, :], thetas,
                   center=centers[height//2],
                   sinogram_order=False,
                   algorithm='art')
    threads = [None] * height
    data_recon = np.zeros((data.shape[1], *r1.squeeze().shape),
                          dtype=np.int16)
    start = perf_counter()
    for i in tqdm(range(height)):
        threads[i] = threading.Thread(target=recon_thread,
                                      args=[i, thetas, centers[i],
                                            data, data_recon,
                                            ],
                                      )
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()
    end = perf_counter()
    print(f'Wall time: {end - start}')
    return data_recon


def recon_thread(idx, thetas, center, arr, arr_out):
    arr_out[idx] = tom.recon(arr[:, idx:idx+1, :],
                             thetas,
                             center=center,
                             sinogram_order=False,
                             algorithm='art').astype(np.int16)


def fbp(viewer: Viewer,
        image: Image,
        COR: bool = False,
        cor_step: int = 100,
        thread: bool = False,
        ):
    original_stack = np.asarray(image.data, dtype=np.int16)
    n_steps, height, _ = original_stack.shape
    thetas = calc_thetas(n_steps)

    if COR:
        print(f'Find COR every {cor_step} pixels vertically')
        center = []
        X = []
        for i in tqdm(range(int(height / cor_step))):
            X.append(i * cor_step)
            center.append(find_center_vo(
                            original_stack[:n_steps//2, :, :],
                            smin=-50, smax=50,
                            ind=i * cor_step,
                            ratio=0.5),
                          )
        print(np.mean(center), center)
        # linear regression
        lm = LinearRegression()
        lm.fit(np.array(X).reshape(-1, 1), center)
        print(f'coeficients: a={lm.coef_[0]}, b={lm.intercept_}.')
        centers = range(height) * lm.coef_[0] + lm.intercept_
        # print('Running FBP with a mean of the CORs')
        # recon = run_fbp(original_stack, thetas, np.mean(center))
        print('Running FBP with  linearly fitted CORs')
        if thread:
            recon = run_fbp_thread(original_stack, thetas, centers)
        else:
            recon = run_fbp(original_stack, thetas, centers)

    else:
        print('Running FBP without COR')
        recon = run_fbp(original_stack, thetas)
    viewer.add_image(recon)
