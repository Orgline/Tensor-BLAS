import ctypes
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

lib = ctypes.CDLL(os.path.join(dir_path, '../build/TensorBLAS.so'))

fun = lib.print_env

# fun.restype = ctypes.c_void_p
# fun.argtypes = []

fun()

syrk = lib.tc_syrk

n = 8
k = 8
alpha = 1.0
beta = 0.0
A = torch.randn(8, 8)
C = torch.randn(8 ,8)
Ah = torch.randn(8, 8, dtype=torch.half)