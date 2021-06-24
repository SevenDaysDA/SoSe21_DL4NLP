import numpy as np
np.random.seed(42)

from tensorflow import keras

from . import Hex10Model
from .data_reader import load_dataset


class Hex10Seq2Seq(Hex10Model):
    def __init__(self, params):
        super(Hex10Seq2Seq, self).__init__(params)
        self.batch_size = params["batch_size"]
        self.hidden_units = params["hidden_units"]
        self.epochs = params["epochs"]


    def train_and_predict(self):
        """
        Trains model on training data. Predicts on the test data.
        :return: Predictions results in the form [(input_1, pred_1, truth_1), (input_2, pred_2, truth_2), ...]
        """

        # Load data: [[token11, token12, ...],[token21,token22,...]]
        # and label: [[label11, label12, ...],[label21,label22,...]]
        X_train_data, y_train_data = load_dataset("train.dat", seq2seq=True)
        X_dev_data, y_dev_data = load_dataset("dev.dat", seq2seq=True)
        X_test_data, y_test_data = load_dataset("test.dat", seq2seq=True)

        latent_dim = 256
        num_encoder_tokens = 71
        num_decoder_tokens = 93
        latent_dim = 256
        num_samples = 10000

        ####################################
        ########  PREPARE DATA   ###########
        ####################################
        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()

        for line in range(min(num_samples, len(X_train_data) - 1)):
            temp_text_x = ''
            temp_text_y = ''
            for char_it in X_train_data[line]:
                temp_text_x += char_it
            for char_it in y_train_data[line]:
                temp_text_y += char_it

            input_text, target_text = temp_text_x ,temp_text_y
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"

            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        print("Number of samples:", len(input_texts))
        print("Number of unique input tokens:", num_encoder_tokens)
        print("Number of unique output tokens:", num_decoder_tokens)
        print("Max sequence length for inputs:", max_encoder_seq_length)
        print("Max sequence length for outputs:", max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        count_err = 0
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            try:
                encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0
            except KeyError:
                pass
                #print("KeyError occured")
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

            try:
                decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0
                decoder_target_data[i, t:, target_token_index[" "]] = 1.0
            except KeyError:
                #print("KeyError occured")
                count_err += 1

        print(f"{count_err} KeyErrors occured")

        ####################################
        ########  BUILD MODEL   ############
        ####################################


        # Define an input sequence and process it.
        encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
        encoder = keras.layers.LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        print("Model created so far")

        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit([encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=self.batch_size,
            epochs= self.epochs,

        )
        # Save model
        model.save("s2s")

        ####################################

        return None
