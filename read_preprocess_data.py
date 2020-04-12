import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE, OneHotEncoder as OHE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
global X
global Y
global musicdata

def oneHotEncoding():
    print("-----------Try to ONE HOT ENCODING-----------------")
    setToCompare = 'abcdefghijklmnopqrstuvwxyz '
    ctoi = dict((c, i) for i, c in enumerate(setToCompare))
    itoc = dict((i, c) for i, c in enumerate(setToCompare))
    # integer encode input data
    integer_encoded = [ctoi[char] for char in musicdata]
    print(integer_encoded)
    # one hot encode
    onehot = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(setToCompare))]
        letter[value] = 1
        onehot.append(letter)
    print(onehot)
    # invert encoding
    inverted = itoc[np.argmax(onehot[0])]
    print(inverted)

    print("--------------------------ENCODING IN PROGRESS----------------------")
    labelencoder = LE()
    X1 = X
    Y1 = Y
    X1[:, 0] = labelencoder.fit_transform(X1[:, 0])
    onehotencoder = OHE([0])
    X1 = onehotencoder.fit_transform(X1)

    labelencoderY = LE()
    Y1 = labelencoderY.fit_transform(Y1)
    print(X1)
    print(Y1)


def getSplitData():
    musicdata = pd.read_csv('sample-data.csv')
    musicdata.head()
    df = pd.DataFrame(musicdata)
    df.plot.box()
    X = musicdata.iloc[:, :-1].values
    Y = musicdata.iloc[:, -1].values
    # print(X)
    # print(Y)

    # print("----duplicates-----------")
    print(musicdata.dtypes)
    duplicate_rows = musicdata[musicdata.duplicated()]
    df = musicdata.drop_duplicates()
    # df.head(5)

    # splitting into train and test
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
    return ((x_train, y_train), (x_test, y_test))


# getSplitData()