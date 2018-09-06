
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io

project_name = 'hayakaawa'
bucket_path = "gs://sklearn-sample"
train_file_path = "{}/dataset/train.csv".format(bucket_path)

# get train data
# CloudML上からgoogle cloud storage上のファイルにアクセスする際には、tensorflowのfile_ioパッケージを使う必要がある
with file_io.FileIO(train_file_path,'r') as f:
    df = pd.read_csv(f)

# train valid split
y = np.array(df['Survived'], dtype='int8')
del df['Survived']
X = np.array(df, dtype='float64')

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# make model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_validation)
acc = accuracy_score(y_validation, y_pred)

print("accuracy: {:.4f}".format(acc))

import os
import pickle

# save model
model_name = 'model.pkl'

with file_io.FileIO(os.path.join(bucket_path,'model',model_name), 'wb') as model_file:
    pickle.dump(clf, model_file)
