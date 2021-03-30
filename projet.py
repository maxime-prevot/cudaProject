from viola_jones import ViolaJones
import time

def bench_train(nbImage):
    start = time.time()

    end = time.time() - start
    pass

def bench_accuracy():
    pass

## Pour éviter les banks conflicts
@cuda.jit(device=True)
def index(index):
    #pour ma cg : GTX 1050 2GB
    nb_banks = 16
    nb_core = 640
    index_div = index // nb_banks
    index_mod = index % nb_banks
    offset_index = ((nb_core * index_mod) + index_div)
    return offset_index

@cuda.jit
def prescan(input, output, n):
    
    ## alloue mémoire partagée
    temp = cuda.shared.array(12288, dtype=float32)
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
    

def transpose(input, output, width, height):
    TPB = 16
    temp = cuda.shared.array(shape=(TPB, TPB+1), dtype=float32)

    xIndex = cuda.blockIdx.x*TPB + cuda.threadIdx.x
    yIndex = cuda.blockIdx.y*TPB + cuda.threadIdx.y

    if xIndex < width and yIndex < height:

        
