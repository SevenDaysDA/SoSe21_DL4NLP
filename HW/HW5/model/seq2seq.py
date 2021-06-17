import numpy as np
np.random.seed(42)

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

        ####################################
        #                                  #
        #   add your implementation here   #
        #                                  #
        ####################################

        return None
