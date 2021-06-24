import numpy as np
np.random.seed(42)

import os

def load_dataset(filename, data_path="data", seq2seq=False):
    inflected_words = []
    lemmata = []

    with open(os.path.join(data_path, filename), 'r') as lines:
        # lists of characters of the inflected word and the lemma
        inflec = []
        lemma = []

        for line in lines:
            # empty line -> a word ends
            if not line.strip():
                if seq2seq:
                    pass
                    ##########################################
                    inflec.append(" ")
                    lemma.append(" ")
                    ##########################################

                # store assembled inputs
                inflected_words.append(inflec)
                lemmata.append(lemma)
                inflec = []
                lemma = []
                continue

            inflec_char, lemma_char = line.strip().split('\t')
            inflec.append(inflec_char)

            if seq2seq:
                pass
                ##########################################

                lemma_char = lemma_char.replace('_MYJOIN_', '').replace('EMPTY', "")
                lemma.append(lemma_char)
                ##########################################
            else:
                lemma.append(lemma_char)
    return inflected_words, lemmata