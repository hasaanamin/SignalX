"""Small bounded FFT-bank microbenchmark, not an acquisition acceptance test."""
import time
import numpy as np
from scipy.fft import fft,ifft


def main():
    rng=np.random.default_rng(2233)
    x=rng.normal(size=4223)+1j*rng.normal(size=4223)
    kernels=rng.normal(size=(27,128))+1j*rng.normal(size=(27,128))
    for workers in (1,2,4):
        for size in (512,1024,2048,4356):
            stride=size-127
            tiles=np.zeros(((4096+stride-1)//stride,size),complex)
            for i,start in enumerate(range(0,4096,stride)):
                region=x[start:min(start+stride,4096)+127]
                tiles[i,:len(region)]=region
            transformed=fft(tiles,axis=-1,workers=1)
            bank=fft(kernels,n=size,axis=-1,workers=1)
            product=transformed[:,None,:]*bank[None,:,:]
            ifft(product,axis=-1,workers=workers) # readiness outside timer
            elapsed=[]
            for repeat in range(15):
                started=time.perf_counter()
                transformed=fft(tiles,axis=-1,workers=1)
                result=ifft(transformed[:,None,:]*bank[None,:,:],axis=-1,workers=workers)
                powers=np.abs(result[:,:,127:])**2
                elapsed.append((time.perf_counter()-started)*1000)
            print(size,workers,np.percentile(elapsed,[50,95,100]).tolist(),flush=True)


if __name__=='__main__': main()
