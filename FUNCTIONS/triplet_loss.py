# References: Ekeany/Siamese-Network-with-Triplet-Loss

import keras
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Input, Lambda
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
print("keras: ", keras.__version__)

import numpy as np
print("numpy: ", np.__version__)

import sklearn
from sklearn.manifold import TSNE
print("sklearn: ", sklearn.__version__)

import matplotlib
import matplotlib.pyplot as plt
print("matplotlib: ", matplotlib.__version__)


class TripletLossModel():
    def __init__(self):
        """
        Loading MNIST data.
        """
        
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def get_image(self, label, test=False):
        """
        Choosing a random image given the label.
        """

        if test:
            X, y = self.X_test, self.y_test
        else:
            X, y = self.X_train, self.y_train
        
        ran_idx = np.random.randint(len(y))
        while y[ran_idx] != label:
            ran_idx = np.random.randint(len(y))

        return X[ran_idx]

    def get_triplet(self, test=False):
        """
        Choosing a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label, but
        the anchor and negative have different labels.
        """

        n = a = np.random.randint(10)
        while n == a: # As long as n == a, run below. If not, get out of the loop.
            n = np.random.randint(10)
        a, p = self.get_image(a, test), self.get_image(a, test)
        n = self.get_image(n, test)

        return a, p, n

    def generate_triplets(self, batch_size=100, test=False):
        """
        Generating an un-ending stream of triplets.
        """

        while True:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batch_size):
                a, p, n = self.get_triplet(test)
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
                
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            # a "dummy" label which will come in to our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batch_size)
            
            yield [A, P, N], label

    @staticmethod
    def identity_loss(y_true, y_pred):
        return K.mean(y_pred)

    @staticmethod
    def triplet_loss(x, alpha = 0.2):
        """
        Calculating a triplet loss.
        """

        anchor, positive, negative = x

        pos_dist = K.sum(K.square(anchor - positive),axis=1)
        neg_dist = K.sum(K.square(anchor - negative),axis=1)

        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss, 0.0)

        return loss

    @staticmethod
    def embed_model():
        """
        Making a simple convolutional model.
        """

        model = Sequential()

        model.add(Convolution2D(32, (3, 3), activation='relu',
                                input_shape=(28, 28, 1)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))

        return model

    def complete_model(self, base_model, imsize=28, LR = 0.0001):
        """
        Adding three embedding models and minimizing the loss among their output embeddings.
        """

        # Create the complete model with three
        # embedding models and minimize the loss 
        # between their output embeddings
        input_1 = Input((imsize, imsize, 1))
        input_2 = Input((imsize, imsize, 1))
        input_3 = Input((imsize, imsize, 1))
            
        A = base_model(input_1)
        P = base_model(input_2)
        N = base_model(input_3)
    
        loss = Lambda(self.triplet_loss)([A, P, N])
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        model.compile(loss=self.identity_loss, optimizer=Adam(LR))

        return model

def plot_model(history):
    """
    Plotting training losses.
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Losses',size = 20)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()