import ctypes
import os

import torch
import sys

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


def tc_trsm(A, B, nb=256):
    # Solve AX = B, A is lower triangular
    # B will be overwritten with X
    # Currently it solves XA^T=B, A is lower triangular

    # Get n, m from A
    n = A.size(0)
    m = B.size(0)
    A = torch.clone(A.T).contiguous()
    B_inout = torch.clone(B.T).contiguous()

    lib.tc_trsm_wrapper.argtypes = [
        ctypes.c_long,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_long
    ]
    lib.tc_trsm_wrapper.restype = ctypes.c_int
    ptr_A = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_float))
    ptr_B = ctypes.cast(B_inout.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Call tc_trsm_wrapper
    status = lib.tc_trsm_wrapper(ctypes.c_long(m), ctypes.c_long(n), ptr_A, ptr_B, ctypes.c_long(nb))
    assert status == 0
    return B_inout.T

def test_tc_trsm(A, B):
    
    C_ref = torch.linalg.solve_triangular(A.T, B, upper=True, left=False)
    #print(C_ref)
    print(C_ref @ A.T-B)
   
    C = tc_trsm(A, B)
    CC = C @ A.T-B
    print(CC)
    # import numpy
    # numpy.savetxt("X.csv", CC.T.cpu(), delimiter=',')
    
    print(f"Error: {torch.linalg.norm(C_ref - C) / torch.linalg.norm(C_ref)}")


if __name__ == '__main__':
    # n = 20000
    # k = 40000
    # A = torch.randn(n, k, device='cuda', dtype=torch.float32)
    # test_tc_syrk(A)
    # tmp=torch.randn(4, 2)
    # print(tmp)
    # tmpt = torch.clone(tmp.T).contiguous()
    # print(tmpt)
    # print(tmpt.reshape(4,2))
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    A = torch.randn(n, n, device='cuda', dtype=torch.float32)
    A = A @ A.t() + torch.eye(n, device='cuda', dtype=torch.float32)
    A = A.tril()
    B = torch.randn(m, n, device='cuda', dtype=torch.float32)
    # A = torch.tensor([[1., 0.], [1., 1.]], device='cuda', dtype=torch.float32)
    # B = torch.tensor([[1., 2.], [4., 5.]], device='cuda', dtype=torch.float32)
    test_tc_trsm(A, B)
