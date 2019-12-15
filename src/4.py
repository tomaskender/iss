from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
import math

obj = []

def calculate_features(filename, invert_axis):
    global obj

    s, fs = sf.read(filename)
    t = np.arange(s.size) / fs

    f, t, sgr = spectrogram(s, fs,nperseg=400, noverlap=240, nfft= 511)
    sfgr_log = 10 * np.log10(sgr+1e-20) 

    matica = []
    for j in range(0, len(sfgr_log), 16):
        riadok = []
        for i in range(len(sfgr_log[j])):
            sum = 0
            for x in range(16):
                sum += sfgr_log[j+x][i]
            riadok.append(sum)
        matica.append(riadok)
    obj = np.array(matica)

    plt.figure(figsize=(9,3))
    plt.pcolormesh(t, range(int(obj.size/obj[0].size)), obj)
    plt.gca().set_title(filename)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvencia [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektrálna hustota výkonu [dB]', rotation=270, labelpad=15)

    if invert_axis:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt

def calculate_score(sentence_matica, sentence_pp, query_matica):
    sentence_matica = np.array(sentence_matica)
    query_matica = np.array(query_matica)
    curr_score = 0.0
    for k in range(len(query_matica.transpose())):
        curr_score += pearsonr(query_matica.transpose()[k], sentence_matica.transpose()[sentence_pp+k])[0]
    return curr_score/len(query_matica[0])


def create_similarity_graph(sentence_matica, query_matica):
    score_mat = []
    for pp in range(len(sentence_matica[0])-len(query_matica[0])):
        score_mat.append(calculate_score(sentence_matica, pp, query_matica))
    plt.figure(figsize=(9,3))
    plt.ylim(0,1)
    plt.plot(np.arange(len(score_mat))/100, score_mat)
    plt.gca().set_title('Priebeh skóre')
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Skóre')

    plt.tight_layout()
    plt.savefig("skore.pdf")
    

calculate_features('./sentences/si512.wav', False).savefig('si512.pdf')
sent_m = obj
calculate_features('./queries/q1.wav', True).savefig('q1.pdf')
query_m = obj

create_similarity_graph(sent_m, query_m)