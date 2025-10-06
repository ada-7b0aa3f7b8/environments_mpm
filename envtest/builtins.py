from typing import List
import pandas as pd
import numpy as np


__all__ = ['rand_array', 'smooth_image', 'my_mat_solve', 'panda_dataframe']


def smooth_image(a, sigma=1):
    return gaussian_filter(a, sigma=sigma)


def rand_array(shape):
    return np.random.rand(*shape)


def my_mat_solve(A, b):
    return A.inv()*b


def panda_dataframe(name: str, nums: List[int]):
    return pd.DataFrame({name: nums})
