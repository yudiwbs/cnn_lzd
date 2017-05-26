#memproses file skor hasil dari predict3_yw
#contoh input per baris: [-19.50937271  21.65632629]
#ganti jadi probabilitas
# HATI_HATI BARIS DUMMY


import re
import numpy as np

#HATI2 nama file
fileInput  = "/home/yudiwbs/lombalazada/data/validasi/run10/score_run10_coba5_concis.csv"
fileOutput = "/home/yudiwbs/lombalazada/data/validasi/run10/prob_run10_coba5_concis.csv"

#fileInput  = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun8/score_run8_clarity.csv"
#fileOutput = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun8/prob_clarity.csv"


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



fInput  = open(fileInput,"r")
fOutput = open(fileOutput, "w")

try:
    i = 0
    for line in fInput:
        line2 = line.rstrip('\n').strip()
        line2 = re.sub('([\[\]])', ' ', line2)
        #buang dulu kurung siku
        angka = line2.split()
        a0 = float(angka[0])
        a1 = float(angka[1])
        scores = []
        scores.append(a0)
        scores.append(a1)
        prob = softmax(scores)
        fOutput.write(str(prob[1])+"\n")
        i = i+1
        print("proses baris ke:"+str(i))
    #endfor
finally:
    fInput.close()
    fOutput.close()