import ctypes
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

lib = ctypes.CDLL(os.path.join(dir_path, '../build/TensorBLAS.so'))

n = 4
k = 8
nb = 1024

print_env = lib.print_env
print_env()

lib.tc_syrk_wrapper.argtypes = [
    ctypes.c_long,
    ctypes.c_long,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_long
]
lib.tc_syrk_wrapper.restype = ctypes.c_int

A = torch.randn(n, k).cuda()
C = torch.zeros(n, n, device='cuda', dtype=torch.float32)
ptr_A = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_float))
ptr_C = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_float))

status = lib.tc_syrk_wrapper(ctypes.c_long(n), ctypes.c_long(k), ptr_A, ptr_C, ctypes.c_long(nb))
assert status == 0
#print(A)

#A = torch.tensor([[1.0, 2, 3], [4, 5, 6]]).cuda()
print(A @ A.t()-C)


