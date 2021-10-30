# -*- coding: utf-8 -*-

import numpy as np

a = np.array([[1, 2], [3, 4]])

b = np.array([[5, 6]])

c = np.concatenate([a, b], axis=0)
print(a)
print(b)
print(c)
