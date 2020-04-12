import pandas as pd
from sklearn.naive_bayes import GaussianNB as GNB
import sklearn.model_selection as skl
from sklearn import preprocessing as ppr
import read_preprocess_data as rpd
from sklearn import metrics as met

# an arbitrary try to Naive Bayes
# musicdata = pd.read_csv('sample-data.csv')
# le = ppr.LabelEncoder()
# x = musicdata.iloc[:, :-1].values
# y = musicdata.iloc[:, -1].values
#
x_train, x_test, y_train, y_test = skl.train_test_split(x, y)
print(x_test)
print(y_test)
(x_train, y_train), (x_test, y_test) = rpd.getSplitData()
model = GNB()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("Model accuracy is : ", met.accuracy_score(y_test,predicted))
