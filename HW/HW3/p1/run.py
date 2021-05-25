# ------------------------------------------------
#             1.1 Data Formats
# ------------------------------------------------

####################################
def read_data(filename, labeled = True ):
    list_scores = []
    list_first_sent = []
    list_second_sent = []
    with open(filename,'r', encoding='utf8') as f:
        for line in f:
            content = line.strip(" ").split('\t')
            list_scores.append(float(content[0]))
            list_first_sent.append(content[1])
            list_second_sent.append(content[2])
    if labeled:
        return list_scores, list_first_sent, list_second_sent
    else:
        return list_first_sent, list_second_sent


print("----------------------------------------------")
print("Excercise 1.1")
print("First sentence pair of Training set:  ")
data = read_data("data-train.txt")

print(data[0][0], data[1][0],data[2][0])


def write_simscore(filename, sim_list):
    with open(filename, 'w') as f:
        for element in sim_list:
            f.write(str(element))
            f.write("\n")
        f.close()
    print("Done writing")

write_simscore("simscores.txt",data[0])
print("----------------------------------------------")
####################################


# ------------------------------------------------
#             1.2 Embedding the Sentences
# ------------------------------------------------
# Expect the embeddings to be accessible at ./wiki-news-300d-1M.vec here, but
# DO NOT upload them to Moodle!
####################################
import  numpy as np


def read_wiki_dic(filename, limit=10000):
    wiki_dic = {}
    try:
        with open(filename, 'r',encoding='utf8') as f:
            counter = 0
            for line in f:
                if counter == 0:
                    counter += 1
                    continue
                element = line.strip('\n').split(' ')
                word = element[0]
                vec  = np.float_(element[1:])
                wiki_dic[word] = vec
                counter += 1
                if counter >= limit+1:
                    break
            f.close()
    except:
        print("Could not read vocab in position {}".format(counter))
    return wiki_dic
wik_dic = read_wiki_dic('wiki-news-300d-1M.vec',limit=40000)

# 1.2 b)
import nltk
from nltk import word_tokenize
nltk.download('punkt')

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

example_sentence = tokenize(read_data("data-train.txt")[1][0])  # Read data überfluessig, da als data defined above

print("Tokenized first sentence of training data set:")
print(example_sentence)
print()

# 1.2 c)
def token_to_vector(token_list):
    vector_list = []
    for token in token_list:
        try:
            vector = wik_dic[token]
        except:
            #print("No words found for {}. Assign zero-vector.".format(token))  #auskommentiert damit man outputs lesen kann
            vector = np.zeros(300)
        vector_list.append(vector)
    return vector_list

# 1.3 d)
def  embed_sentence(vector_list):
    sum_vector = np.zeros(300)
    for vec in vector_list:
        sum_vector += vec
    return np.divide(sum_vector, len(vector_list))

print("First 20 dimensions of averaged vector:")
print(embed_sentence(token_to_vector(example_sentence))[:20])
####################################
# ------------------------------------------------
#             1.3 Scoring the Similarity
# ------------------------------------------------

####################################
from keras.models import Sequential, Model
from keras.layers import *
import tensorflow as tf
from tensorflow import keras
np.random.seed(7)

# 2b) creating the model

# 2 Inputs, one for each sentence embedding
input1 = Input(shape=(300,))
input2 = Input(shape=(300,))

# A layer that concatenates both inputs
inputs = concatenate(axis=1, inputs=[input1, input2])

# A dropout layer with probability 0.3
layer1 = Dropout(0.3)
x = layer1(inputs)

# A Dense layer with 300 dimensions and relu activation
x = Dense(300, activation='relu')(x)

# A dropout layer with probability 0.3
x = Dropout(0.3)(x)

# A Dense layer with 1 dimension and sigmoid activation
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input1, input2], outputs=outputs)

print(model.summary())

# 2c) Compile the Model with Adam and MSE

model.compile(optimizer="Adam", loss="mse", metrics=["mse"])

# 2d) Train the model, Batch size = 100, epochs= 300

# formatting

y_train = np.asarray(data[0])    # Evaluations

temp1 = []
for i in data[1]:
    temp1.append(embed_sentence(token_to_vector(i)))


temp2 = []
for i in data[2]:
    temp2.append(embed_sentence(token_to_vector(i)))

x_train = [np.asarray(temp1), np.asarray(temp2)]

# 2d)

history = model.fit(x=x_train, y=y_train, batch_size=100, epochs=300)
print("Fit model on training data")

# MSE = 0.0626 after 300 epochs on train set

history.history

# Evaluate model on development data set

data_dev = read_data("data-dev.txt")

y_dev = np.asarray(data_dev[0])    # Evaluations

temp1 = []
for i in data_dev[1]:
    temp1.append(embed_sentence(token_to_vector(i)))


temp2 = []
for i in data_dev[2]:
    temp2.append(embed_sentence(token_to_vector(i)))

x_dev = [np.asarray(temp1), np.asarray(temp2)]

print("Finished reading Dev. Data")


results = model.evaluate(x_dev, y_dev, batch_size=100)

print("Final mean squared error on development data set:", results[0])


####################################
