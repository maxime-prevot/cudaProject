
from numba import cuda
import numba as nb

@cuda.jit
def test():
    print("yes")


test[1,1]()

test[1,33]()
cuda.synchronize()

def scan(input, outpout, n):
    temp = []
    tdx = cuda.threadIdx.x
    offset = 1

    temp[2*tdx] = input[2*tdx]
    temp[2*tdx+1] = input[2*tdx+1]


def transpose():

