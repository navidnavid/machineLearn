from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression

class DecisionTree:
    def __init__(self, x, y, columns):
        self.to_leran=x
        self.result=y
        self.columns = columns
        self.test_size=0.3
        self.verbos=0
        self.criterion='entropy'
        self.accuracy_score_ =0
        self.dtc_ = []


    def get_and_leran_1d(self):
        x = self.to_leran
        y = self.result
        random_state=0
        df = pd.DataFrame(x, columns=self.columns)
        df["Target"] = y
        #print(df)
        X = df.drop("Target", axis=1)
        #print(X)
        y = df["Target"]
        #clf = DecisionTreeClassifier()
        #clf.fit(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=random_state, test_size=self.test_size)
        if self.verbos==1:
            print(" train data")
            print(X_train)
            print("---")
            print(y_train)
            print("---")
            print(X_test)
            print("---")
            print(y_test)

        dtc = DecisionTreeClassifier(criterion=self.criterion)
        dtc.get_params()
        clf=dtc.fit(X_train, y_train)
        y_pred=dtc.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        self.accuracy_score_ = accuracy_score(y_test, y_pred)
        self.dtc_ = dtc

    def Predict(self , X_test):
        y_pred = self.dtc_.predict(X_test)
        return y_pred

# #-----------------------------

#     df = pd.DataFrame(x, columns=["Feature 1", "Feature 2"])
#     df["Target"] = y

#     #print(df)
#     X = df.drop("Target", axis=1)
#     #print(X)
#     y = df["Target"]

#     clf = DecisionTreeClassifier()
#     clf.fit(X, y)

#     X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.3)
#     print(" train data")
#     print(X_train)
#     print("---")
#     print(y_train)
#     print("---")
#     print(X_test)
#     print("---")
#     print(y_test)
#     dtc = DecisionTreeClassifier(criterion='entropy')
#     dtc.get_params()
#     clf=dtc.fit(X_train,y_train)
#     y_pred=dtc.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     tree.plot_tree(clf)
