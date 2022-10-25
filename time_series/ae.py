'''
Use Autoencoder for time series classification
'''

from pathlib import Path
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def lstm(X_train, X_test, y_train, y_test):
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    # TO-DO: rewrite input layer
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

def lstm_cnn(X_train, X_test, y_train, y_test):

    # fix random seed for reproducibility
    #numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    #top_words = 5000
    #(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    # truncate and pad input sequences
    #max_review_length = 500
    #X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    #X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    # TO-DO: rewrite input layer
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

if __name__ == '__main__':
    fn = 'analog_dataset_Xy_rtus-22-26-172-228-230-11024-11027_alltime.csv'
    fp = Path(Path().absolute().parents[1], 'notebooks', fn)
    print(fp)
    if fp.is_file():
        # import dataset
        df = pd.read_csv(fp).iloc[:, 1:]
        print('dataframe dimensions: ', df.shape)
        X = df.drop(['Label'], axis=1)
        y = df['Label']
        print('X: ', X.shape, 'y: ', y.shape)

        X_scaled = MinMaxScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=2021)
        lstm_cnn(X_train, X_test, y_train, y_test)