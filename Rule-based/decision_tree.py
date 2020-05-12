import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import Preprocess.make_dataset as md
import Preprocess.csv_utils as cu
import Preprocess.meta_data_2018 as md8

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = cu.drop_attribute(df, ['Timestamp'])
    df = df.dropna(axis=0)
    df = md.convert_datatype(df, "ids2018")
    df = md.label_encoding(df, md8.LABEL_LIST, 1, "ids2018")
    df = shuffle(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)
    print(df.shape)
    print(df.isnull().values.nany())
    values = df.values
    X, Y = values[:, :-1], values[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)

    Y_train_pred = dt.predict(X_train)
    Y_test_pred = dt.predict(X_test)
    print("Train Accuracy: ", accuracy_score(Y_train, Y_train_pred))
    print("Test Accuracy: ", accuracy_score(Y_test, Y_test_pred))

if __name__ == "__main__":
    train("../../10percent/10percent.csv")