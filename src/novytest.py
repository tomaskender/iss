from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython

#sf.read('./queries/q1.wav')
#sf.read('./queries/q2.wav')

#for pre kazdu sentence
s, fs = sf.read('./sentences/sa1.wav')
t = np.arange(s.size) / fs

#f, t, sgr = spectrogram(s, fs,nperseg=400, noverlap=240, nfft= 511)
#sfgr_log = 10 * np.log10(sgr+1e-20) 

sfgr_log = [ [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6] ]

matica = []
for i in range(len(sfgr_log[0])):
    riadok = []
    #po riadkoch
    for j in range(0, len(sfgr_log), 3):
        sum = 0
        for x in range(3):
            sum += sfgr_log[j+x][i]
        riadok.append(sum)
    matica.append(riadok)

npObj = np.array(matica)





plt.figure(figsize=(9,3))
plt.pcolormesh(6, 2, npObj.transpose())
plt.gca().set_title('sa1.wav')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()

plt.savefig('yourfile.pdf')
#IPython.display.display(IPython.display.Audio(s, rate=fs))