from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython


s, fs = sf.read('./sentences/sa1.wav')
s = s[:113323]
t = np.arange(s.size) / fs

f, t, sgr = spectrogram(s, fs,nperseg=400, noverlap=240, nfft= 511)
sfgr_log = 10 * np.log10(sgr+1e-20) 

plt.figure(figsize=(9,3))
plt.pcolormesh(t,f,sfgr_log)
plt.gca().set_title('sa1.wav')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()

plt.savefig('yourfile.pdf')
#IPython.display.display(IPython.display.Audio(s, rate=fs))