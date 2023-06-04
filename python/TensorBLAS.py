import ctypes
import os

import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

lib = ctypes.CDLL(os.path.join(dir_path, '../build/TensorBLAS.so'))

def tc_syrk(A, transpose=False, nb=256):
    if transpose:
        A = torch.clone(A.T).contiguous()

    # Get n, k from A
    n = A.size(0)
    k = A.size(1)
    C = torch.zeros(n, n, device=A.device, dtype=torch.float32)

    lib.tc_syrk_wrapper.argtypes = [
        ctypes.c_long,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_long
    ]
    lib.tc_syrk_wrapper.restype = ctypes.c_int
    ptr_A = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_float))
    ptr_C = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Call tc_syrk_wrapper
    status = lib.tc_syrk_wrapper(ctypes.c_long(n), ctypes.c_long(k), ptr_A, ptr_C, ctypes.c_long(nb))
    assert status == 0
    return C

def test_tc_syrk(A):
    C = tc_syrk(A)
    print(f"Error: {torch.linalg.norm(A @ A.t()-C)/torch.linalg.norm(C)}")
    C = tc_syrk(A, transpose=True)
    print(f"Transpose Error: {torch.linalg.norm(A.t() @ A-C)/torch.linalg.norm(C)}")


if __name__ == '__main__':
    n = 20000
    k = 40000
    A = torch.randn(n, k, device='cuda', dtype=torch.float32)
    test_tc_syrk(A)
