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


        num_encoder_tokens = 81
        num_decoder_tokens = 93
        latent_dim = self.hidden_units
        num_samples = 100000

        ####################################
        ################ Prepare Test Set
        ####################################
        test_input_texts = []
        test_target_texts = []
        test_input_characters = set()
        test_target_characters = set()
        for line in range(min(num_samples, len(X_test_data) - 1)):
            temp_text_x = ''
            temp_text_y = ''
            for char_it in X_test_data[line]:
                temp_text_x += char_it
            for char_it in y_test_data[line]:
                temp_text_y += char_it

            input_text = temp_text_x
            target_text =temp_text_y
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"


            test_input_texts.append(input_text)
            test_target_texts.append(target_text)
            for char in input_text:
                if char not in test_input_characters:
                    test_input_characters.add(char)
            for char in target_text:
                if char not in test_target_characters:
                    test_target_characters.add(char)
        test_input_characters = sorted(list(test_input_characters))
        test_target_characters = sorted(list(test_target_characters))
        test_num_encoder_tokens = len(test_input_characters)
        test_num_decoder_tokens = len(test_target_characters)
        test_max_encoder_seq_length = max([len(txt) for txt in test_input_texts])
        test_max_decoder_seq_length = max([len(txt) for txt in test_target_texts])



        test_input_token_index = dict([(char, i) for i, char in enumerate(test_input_characters)])
        test_target_token_index = dict([(char, i) for i, char in enumerate(test_target_characters)])


        test_encoder_input_data = np.zeros(
            (len(test_input_texts), test_max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        test_decoder_input_data = np.zeros(
            (len(test_input_texts), test_max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        test_decoder_target_data = np.zeros(
            (len(test_input_texts), test_max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        for i, (input_text, target_text) in enumerate(zip(test_input_texts, test_target_texts)):
            for t, char in enumerate(input_text):
                test_encoder_input_data[i, t, test_input_token_index[char]] = 1.0
            #print(encoder_input_data.shape)
            test_encoder_input_data[i, t + 1:, test_input_token_index[" "]] = 1.0

            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                test_decoder_input_data[i, t, test_target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    test_decoder_target_data[i, t - 1, test_target_token_index[char]] = 1.0

            test_decoder_input_data[i, t + 1:, test_target_token_index[" "]] = 1.0
            test_decoder_target_data[i, t:, test_target_token_index[" "]] = 1.0

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

            input_text = temp_text_x
            target_text =temp_text_y
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
        #num_encoder_tokens = len(input_characters)
        #num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        print("Number of samples:", len(input_texts))
        print("Number of unique input tokens:", num_encoder_tokens)
        print("Number of unique output tokens:", num_decoder_tokens)
        print("Max sequence length for inputs:", max_encoder_seq_length)
        print("Max sequence length for outputs:", max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
        print(input_token_index[" "])

        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            #print(encoder_input_data.shape)
            encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0

            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

            decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0
            decoder_target_data[i, t:, target_token_index[" "]] = 1.0


        ####################################
        ########  BUILD MODEL   ############
        ####################################
        model = keras.models.load_model("s2s")
        if not model:
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
                epochs= self.epochs

            )
            # Save model
            model.save("s2s")

        # Define sampling models
        # Restore the model and construct the encoder and decoder.
        #model = keras.models.load_model("s2s")

        encoder_inputs = model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
        decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, target_token_index["\t"]] = 1.0

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ""
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                #print(sampled_token_index)
                sampled_char = reverse_target_char_index[sampled_token_index]
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1.0

                # Update states
                states_value = [h, c]
            return decoded_sentence

        prediction = []

        for seq_index in range(len(test_encoder_input_data)):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = test_encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(input_seq)
            '''
            print("-")
            print("Input sentence:", input_texts[seq_index])
            print("Decoded sentence:", decoded_sentence)
            '''
            pred_element = test_input_texts[seq_index].strip(), decoded_sentence.strip(), test_target_texts[seq_index].strip().replace("\\","").replace("\n","")
            #print(pred_element)
            prediction.append(pred_element)
        print("done")



        ####################################
        print("Size Test set = ", len(y_test_data))
        return prediction
