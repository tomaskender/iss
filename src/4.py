from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
import math

matica = []

def calculate_features(filename):
    global matica

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
    plt.pcolormesh(range(obj[0].size), range(int(obj.size/obj[0].size)), obj)
    plt.gca().set_title('sa1.wav')
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt

def calculate_score(sentence_matica, sentence_pp, query_matica):
    sentence_matica = np.array(sentence_matica)
    query_matica = np.array(query_matica)
    curr_score = 0
    for k in range(len(query_matica[0])):
        val = pearsonr(sentence_matica.transpose()[sentence_pp+k], query_matica.transpose()[k])[1]
        if math.isnan(val):
            val = 0
        curr_score += val 
    return curr_score/len(query_matica[0])


def create_similarity_graph(sentence_matica, query_matica):
    score_mat = []
    for pp in range(len(sentence_matica[0])-len(query_matica[0])):
        score_mat.append(calculate_score(sentence_matica, pp, query_matica))
    plt.figure(figsize=(9,3))
    print(score_mat)
    #plt.plot(np.linspace(0, len(sentence_matica[0])), np.array(score_mat))
    plt.gca().set_title('Priebeh skore.wav')
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Skore')

    plt.tight_layout()
    plt.savefig("skore.pdf")
    

calculate_features('./sentences/si512.wav')#.savefig('output.pdf')
sent_m = matica
calculate_features('./queries/q1.wav')
query_m = matica

create_similarity_graph(sent_m, query_m)