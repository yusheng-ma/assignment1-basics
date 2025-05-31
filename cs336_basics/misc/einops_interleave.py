import numpy as np
from einops import rearrange

a = np.array([[1, 2], [3, 4], [5, 6]])

ans_a = [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]]

try_a = rearrange([a, a], 't h w -> h (w t)')

print(try_a)