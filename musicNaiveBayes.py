import pandas as pd
from sklearn.naive_bayes import GaussianNB as GNB
import sklearn.model_selection as skl
from sklearn import preprocessing as ppr
import read_preprocess_data as rpd
from sklearn import metrics as met

# an arbitrary try to Naive Bayes
musicdata = pd.read_csv('sample-data.csv')
le = ppr.LabelEncoder()
# first performing encoding to the dependent and independant columns as it is categorical
#  - unfortunately, the encoding part is not functioning right...
x = musicdata.iloc[:, :-1].values
y = musicdata.iloc[:, -1].values
# x = le.fit_transform(x)
# y = le.fit_transform(y)
# split into tuples of train and test
# x_train, x_test, y_train, y_test = skl.train_test_split(x, y)
# print(x_test)
# print(y_test)

# invoking get split data of our global file, read_preprocess_data
(x_train, y_train), (x_test, y_test) = rpd.getSplitData()
# invoking the Naive Bayes Function
model = GNB()

# fit the training set
model.fit(x_train, y_train)
# predict the testset.
predicted = model.predict(x_test)
print("Model accuracy is : ", met.accuracy_score(y_test,predicted))
