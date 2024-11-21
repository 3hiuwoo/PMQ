import torch
import numpy as np
import pywt
from scipy.interpolate import interp1d
from .utils import get_device


def normalize(arr):
    '''
    normalize the array by x = (x - x.min()) / (x.max() - x.min())
    '''
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


class ToTensor:
    '''
    convert ndarrays to tensor and cast to float32
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, signal):
        signal = torch.from_numpy(signal)
        
        return signal.to(torch.float32)
    

class Compose:
    '''
    apply sequential transformation on the signal
    '''
    def __init__(self, transforms):
        self.transforms = transforms
        
    
    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal
    

class Normalize:
    '''
    normalize the signal by x = (x - x.min()) / (x.max() - x.min())
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, signal):
        return normalize(signal)
    
    
class Scale:
    '''
    scale the signal
    
    Args:
        sf(int/float): scaling factor
    '''
    def __init__(self, sf):
        self.sf = sf
    
    
    def __call__(self, signal):
        return signal * self.sf


class VerticalFlip:
    '''
    negate the signal
    
    Args:
        norm(bool, optional): normalize the signal after negating
    '''
    def __init__(self, norm=False):
        self.norm = norm
    
    
    def __call__(self, signal):
        return (normalize(-signal) if self.norm else -signal)


class HorizontalFlip:
    '''
    invert the signal temporally
    
    note:
        cannot receive tensor as parameter
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, signal):
        return np.flip(signal, axis=-1)


class AddNoise:
    '''
    add noise to the signal
    
    Args:
        snr(int/float): signal to noise ratio
        
    note:
        the dtype of the return tensor will be forced to float64
    '''
    def __init__(self, snr):
        self.snr = snr
    
    
    def __call__(self, signal):
        noise = np.random.normal(0, self._get_std(signal, self.snr), signal.shape)
        return signal + noise
    
    
    def _get_std(self, arr, snr):
        avg_power_signal = (arr ** 2).mean()
        avg_power_noise = 10 ** ((avg_power_signal - snr) / 10)
        return (avg_power_noise ** 0.5)
    
    
class Permute:
    '''
    permute the signal
    
    Args:
        n(int): number of segments to be divided into
        
    note:
        will return ndarray when receive tensor as parameter
    '''
    def __init__(self, n):
        self.n = n
    
    
    def __call__(self, signal):
        segs = np.array_split(signal, self.n, axis=-1)
        np.random.shuffle(segs)
        return np.concatenate(segs, axis=-1)
    
    
class TimeWarp:
    '''
    warp the signal in time
    
    Args:
        n(int): number of segments to be divided into
        sf(int/float): stretch factor(>1) or squeeze factor(<1)
        
    note:
        will return ndarray when receive tensor as parameter
    '''
    def __init__(self, n, sf):
        self.n = n
        self.sf = sf
        
        
    def __call__(self, signal):
        segs = np.array_split(signal, self.n, axis=-1)
        choices = np.random.choice(self.n, self.n//2, replace=False)
        choices.sort()
        
        # stretch/squeeze selected signal
        for i in range(self.n):
            if i in choices:
                segs[i] = self._warp(segs[i], self.sf)
            else:
                segs[i] = self._warp(segs[i], 1/self.sf)
                
        warp_signal = np.concatenate(segs, axis=-1)
        if warp_signal.shape[-1] < signal.shape[-1]:
            warp_signal = np.pad(warp_signal,
                ([(0, 0)] * (warp_signal.ndim - 1) +
                [(0, signal.shape[-1] - warp_signal.shape[-1])]))
        elif warp_signal.shape[-1] > signal.shape[-1]:
            warp_signal = warp_signal[..., :signal.shape[-1]]
        return warp_signal
        
        
    def _warp(self, signal, sf):
        x_old = np.linspace(0, 1, signal.shape[-1])
        x_new = np.linspace(0, 1, int(signal.shape[-1] * sf))
        f = interp1d(x_old, signal, axis=-1)
        return f(x_new)
        
    
class ToGroup:
    '''
    convert the signal to a group of signals with different transformations
    
    Args:
        trans(array): list of transformations
    '''
    def __init__(self, transforms):
        self.transforms = transforms
        

    def __call__(self, signal):
        sigs = [signal]
        for t in self.transformations:
            sigs.append(t(signal))
        return np.concatenate(sigs, axis=0)
    

class WaveletDenoise:
    '''
    denoise the raw signal
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, signal):
        coeffs = pywt.wavedec(data=signal, wavelet='db5', level=9)
        cD2, cD1 = coeffs[-2], coeffs[-1]

        threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
        cD1.fill(0)
        cD2.fill(0)
        for i in range(1, len(coeffs) - 2):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)

        sig = pywt.waverec(coeffs=coeffs, wavelet='db5')
        return sig
        

class Segment:
    '''
    segment the signal sequentially
    '''
    def __init__(self, length=2500):
        self.idx = 0
        self.length = length
        
        
    def __call__(self, signal):
        i, l = self.idx, self.length
        if i*l >= signal.shape[-1] or (i+1)*l > signal.shape[-1]:
            i = 0
            self.idx = 0
        self.idx += 1
        return signal[..., i*l:(i+1)*l]  
    
    
class CreateView:
    '''
    create multiple views used for contrastive learning
    '''
    def __init__(self, transform, nviews=2):
        self.transform = transform
        self.nviews = nviews
        
        
    def __call__(self, signal):
        signals = [self.transform(signal) for n in range(self.nviews)]
        return np.stack(signals, axis=0)
        

class DownSample:
    '''
    downsample the signal
    
    Args:
        df(int): downsample factor
    '''
    def __init__(self, df):
        self.df = df
        
        
    def __call__(self, signal):
        return signal[..., ::self.df]