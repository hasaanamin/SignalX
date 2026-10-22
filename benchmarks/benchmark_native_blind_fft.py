"""Isolated FFT-kernel experiment; not selected by production code."""
import time
import numpy as np
from numba import njit,literal_unroll
from scipy.fft import fft,ifft


@njit(cache=True,nogil=True)
def transform(values,twiddles,reverse):
    for k in range(512):
        j=reverse[k]
        if j>k:
            temp=values[k];values[k]=values[j];values[j]=temp
    length=2
    while length<=512:
        half=length//2;step=512//length
        for base in range(0,512,length):
            for j in range(half):
                a=values[base+j]
                b=values[base+j+half]*twiddles[j*step]
                values[base+j]=a+b;values[base+j+half]=a-b
        length*=2


@njit(cache=True,nogil=True)
def bank_powers(samples,kernels,reverse,forward,inverse,count):
    powers=np.empty((len(kernels),count))
    spectrum=np.empty(512,np.complex128)
    work=np.empty(512,np.complex128)
    for offset in range(0,count,385):
        n=min(385,count-offset)
        for j in range(512): spectrum[j]=samples[offset+j] if j<n+127 else 0j
        transform(spectrum,forward,reverse)
        for branch in range(len(kernels)):
            for j in range(512): work[j]=spectrum[j]*kernels[branch,j]
            transform(work,inverse,reverse)
            for j in range(n):
                value=work[127+j]/512
                powers[branch,offset+j]=value.real*value.real+value.imag*value.imag
    return powers


@njit(cache=True,nogil=True,inline='always')
def stages(values,twiddles):
    for length in literal_unroll((2,4,8,16,32,64,128,256,512)):
        half=length//2;step=512//length
        for base in range(0,512,length):
            for j in range(half):
                a=values[base+j]
                b=values[base+j+half]*twiddles[j*step]
                values[base+j]=a+b;values[base+j+half]=a-b


@njit(cache=True,nogil=True)
def bank_powers_fast(samples,kernels,reverse,forward,inverse,count):
    powers=np.empty((len(kernels),count))
    spectrum=np.empty(512,np.complex128)
    work=np.empty(512,np.complex128)
    for offset in range(0,count,385):
        n=min(385,count-offset)
        for j in range(512):
            r=reverse[j]
            spectrum[j]=samples[offset+r] if r<n+127 else 0j
        stages(spectrum,forward)
        for branch in range(len(kernels)):
            for j in range(512):
                r=reverse[j]
                work[j]=spectrum[r]*kernels[branch,r]
            stages(work,inverse)
            for j in range(n):
                value=work[127+j]/512
                powers[branch,offset+j]=value.real*value.real+value.imag*value.imag
    return powers


def main():
    rng=np.random.default_rng(2233)
    samples=rng.normal(size=4223)+1j*rng.normal(size=4223)
    kernels=rng.normal(size=(27,128))+1j*rng.normal(size=(27,128))
    bank=fft(kernels,n=512,axis=-1,workers=1)
    reverse=np.array([int(f'{k:09b}'[::-1],2) for k in range(512)])
    forward=np.exp(-2j*np.pi*np.arange(256)/512);inverse=forward.conj()
    actual=bank_powers(samples,bank,reverse,forward,inverse,4096)
    expected=np.abs(np.array([np.convolve(samples,row)[127:127+4096] for row in kernels]))**2
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(bank_powers_fast(samples,bank,reverse,forward,inverse,4096),expected,rtol=1e-11,atol=1e-12)
    for native in (0,1,2):
        elapsed=[]
        for repeat in range(10):
            started=time.perf_counter()
            if native==2: bank_powers_fast(samples,bank,reverse,forward,inverse,4096)
            elif native==1: bank_powers(samples,bank,reverse,forward,inverse,4096)
            else:
                powers=np.empty((27,4096))
                for offset in range(0,4096,385):
                    n=min(385,4096-offset)
                    transformed=fft(samples[offset:offset+n+127],n=512,workers=1)
                    powers[:,offset:offset+n]=np.abs(ifft(bank*transformed,axis=-1,workers=1)[:,127:127+n])**2
            elapsed.append((time.perf_counter()-started)*1000)
        print(('scipy','native','native_static_stages')[native],np.percentile(elapsed,[50,95,100]).tolist())


if __name__=='__main__': main()
