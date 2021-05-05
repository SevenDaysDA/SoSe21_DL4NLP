from gensim.models import KeyedVectors
import numpy as np
import scipy

def Read_SimLex():
    word1_list, word2_list, simlex999_list = [], [], []
    with open("SimLex-999/SimLex-999.txt","r") as f:
        simlex_data = f.readlines()
        for row in range(1,len(simlex_data)):
            e_list = simlex_data[row].split("\t")
            word1_list.append(e_list[0])
            word2_list.append(e_list[1])
            simlex999_list.append(float(e_list[3]))
        f.close()
    return word1_list,word2_list,simlex999_list

# Euclidean Distance
def Euc_Dis(a,b):
    sum_arg = 0
    for i in range(len(a)):
        sum_arg += np.power(a[i]-b[i] , 2 )
    return np.sqrt(sum_arg)




print("Reading pre-trained dataset from Google.")
wv_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True,limit=500000)
print("Done reading.")

w1_list, w2_list, sim_list = Read_SimLex()
euclidean_list = []

words_not_found = []
print("Calculate eucliden distance")
for j in range(len(w1_list)):
    try:
        vec1 = wv_from_bin[w1_list[j]]
    except:
        words_not_found.append(w1_list[j])
        vec1 = np.zeros(300)
    try:
        vec2 = wv_from_bin[w2_list[j]]
    except:
        words_not_found.append(w2_list[j])
        vec2 = np.zeros(300)
    euclidean_list.append(Euc_Dis(vec1,vec2))
print("Done calculating")

print(words_not_found)

value = scipy.stats.pearsonr(sim_list,euclidean_list)
print(value)