
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model  import LinearRegression

from learn import DecisionTree

#data = [[1, 'A'], [2, 'B'], [3, 'C']] 
## Create a DataFrame with column names
#df = pd.DataFrame(data, columns=['ID', 'Category'])

# Defining main function
def main():
    
    x = [[1, 2], [3, 4], [5, 6]]
    y = [-1, -2, -3]
    dt_obj =  DecisionTree(x ,y, columns=["F1", "2"] )
    dt_obj.get_and_leran_1d()
    x_pred= dt_obj.Predict(x)
    print(x_pred)

# Using the special variable 
# __name__
if __name__=="__main__":
    main()


