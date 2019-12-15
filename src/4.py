from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy.stats import pearsonr
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
import math


obj = []

def calculate_features(filename, is_query):
    global obj

    folder = ''
    if is_query:
        folder = './queries/'
    else:
        folder = './sentences/'

    s, fs = sf.read(folder + filename)
    t = np.arange(s.size) / fs

    plt.figure(figsize=(9,3))
    plt.plot(t, s)
    #plt.gca().set_title('Audio signal of ' + filename)
    plt.gca().set_xlabel('t')
    plt.margins(x=0)
    plt.savefig('signal_' + filename + '.pdf')

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
    #plt.gca().set_title(filename)
    plt.gca().set_xlabel('t')
    plt.gca().set_ylabel('Features')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('spectro_' + filename + '.pdf')

def calculate_score(sentence_matica, sentence_pp, query_matica):
    sentence_matica = np.array(sentence_matica)
    query_matica = np.array(query_matica)
    curr_score = 0.0
    for k in range(len(query_matica.transpose())):
        curr_score += pearsonr(query_matica.transpose()[k], sentence_matica.transpose()[sentence_pp+k])[0]
    return curr_score/len(query_matica[0])


def calculate_score_mat(sentence_matica, query_matica):
    score_mat = []
    for pp in range(len(sentence_matica[0])-len(query_matica[0])):
        score_mat.append(calculate_score(sentence_matica, pp, query_matica))
    return score_mat
    
def create_similarity_graph(filename):
    calculate_features(filename, False)
    sentence_matica = obj
    
    calculate_features('q1.wav', True)
    q1_score = calculate_score_mat(sentence_matica, obj)
    calculate_features('q2.wav', True)
    q2_score = calculate_score_mat(sentence_matica, obj)

    plt.figure(figsize=(9,3))
    plt.ylim(0,1)
    plt.margins(x=0)
    plt.plot(np.arange(len(q1_score))/100, q1_score, label='effective')
    plt.plot(np.arange(len(q2_score))/100, q2_score, label='appreciated')
    plt.legend(loc='upper right')
    #plt.gca().set_title('effective and appreciated vs ' + filename)
    plt.gca().set_xlabel('t')
    plt.gca().set_ylabel('Scores')
    plt.tight_layout()
    plt.savefig('score_' + filename + '.pdf')

def create_score_graphs():
    for file in listdir('./sentences/'):
        create_similarity_graph(file)

create_score_graphs()