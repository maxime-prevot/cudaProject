from viola_jones import ViolaJones
import time
from numba import cuda
import numba as nb
import numpy as np
import math
from viola_jones import ViolaJones
import pickle

def bench_train(nbImage):
    start = time.time()

    end = time.time() - start
    pass

def bench_accuracy():
    pass


## Pour éviter les banks conflicts
@cuda.jit
def index(index):
    #pour ma cg : GTX 1050 2GB
    nb_banks = 16
    nb_core = 640
    index_div = index // nb_banks
    index_mod = index % nb_banks
    offset_index = ((nb_core * index_mod) + index_div)
    return offset_index

@nb.njit
def prescan(input, output, n):
    
    ## alloue mémoire partagée
    temp = cuda.shared.array(12288, dtype=nb.types.float64)
    tdx = cuda.threadIdx.x
    threadblocks = cuda.blockIdx.x*cuda.blockDim.x
    offset = 1

    ai = tdx
    bi = tdx + (n//2)

    ## on crée des shifted index afin d'éviter les banks conflicts
    shifted_ai = index(ai)
    shifted_bi = index(bi)

     ##on charge l'input dans la mémoire partagée
    temp[shifted_ai] = input[ai + threadblocks]
    temp[shifted_bi] = input[bi + threadblocks]

    #phase montante
    d = n//2
    while d > 0:
        cuda.syncthreads()
        if tdx < d:
            ai = offset*(2*tdx+1)-1
            bi = offset*(2*tdx+2)-1
            shifted_ai = index(ai)
            shifted_bi = index(bi)
            temp[shifted_bi] += temp[shifted_ai]
        offset *= 2
        #on shift d sur la droite
        d >>= 1
    cuda.syncthreads()

    ## on clear le dernier element
    if tdx == 0:
         temp[index(n-1)] = 0
    
    
    #phase descendante 
    d = 1
    while d < n:
        offset >>=1
        cuda.syncthreads()

        if tdx < d:
            ai = offset*(2*tdx+1)-1
            bi = offset*(2*tdx+2)-1
            shifted_ai = index(ai)
            shifted_bi = index(bi)
            t = temp[shifted_ai]
            temp[shifted_ai] = temp[shifted_bi]
            temp[shifted_bi] += t

        d *=2
    cuda.syncthreads()


    #on écrit les resultats sur la mémoire
    output[ai + threadblocks] = temp[shifted_ai]
    output[bi + threadblocks] = temp[shifted_bi]

    cuda.syncthreads()
    
@nb.njit
def transpose(input, output, width, height):
    TPB = 16
    temp = cuda.shared.array(shape=(TPB, TPB+1), dtype=nb.types.float64)

    xIndex = cuda.blockIdx.x*TPB + cuda.threadIdx.x
    yIndex = cuda.blockIdx.y*TPB + cuda.threadIdx.y

    if xIndex < width and yIndex < height:
        id_input = yIndex * width + xIndex
        temp[cuda.threadIdx.y][cuda.threadIdx.x] = input[id_input]

    cuda.syncthreads()

    xIndex = cuda.blockIdx.x*TPB + cuda.threadIdx.x
    yIndex = cuda.blockIdx.y*TPB + cuda.threadIdx.y
    if xIndex * height and yIndex * width:
        id_output = yIndex * height + xIndex
        output[id_output] = temp[cuda.threadIdx.x][cuda.threadIdx.y]


@cuda.jit
def integral_image(image, output, output_transpose,output_final):
    input = image
    n = len(image)
    prescan(input,output,n)
    transpose(output,output_transpose,np.size(image,0),np.size(image,1))
    prescan(output_transpose,output_final,len(output_transpose))
    return output
    
## essai de calcul d'image integrale ...
with open("training.pkl", 'rb') as f:
    training = pickle.load(f)
    output = np.zeros(training[0][0].shape)
    output_transpose = np.zeros(training[0][0].shape)
    output_final = np.zeros(training[0][0].shape)
    integral_image[16,16](training[0][0],output,output_transpose,output_final)

