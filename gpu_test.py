import numpy as np
import cupy as cp


x_gpu = cp.zeros(10000000)
print("GPU Array:", x_gpu)
