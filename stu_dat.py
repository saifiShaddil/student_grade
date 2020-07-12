import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plot
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1","G2","G3","absences","failures","studytime","freetime"]]

# to print first five row from dataset
# print (data.head()) 

predict  = "G3"

X = np.array(data.drop([predict], 1)) # all attribute
y = np.array(data[predict])           # all labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

"""best  = 0
for _ in range(30):
# taking all the labels and attribute to train them in our model
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
    # saving the best possible model usinf pickle 
    with open ("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f)  # taking linear model and save it in f """

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
    
# they are the value of coefficent of m from y=mx+b
print ("Co: \n", linear.coef_) 
print ("Intercept: \n", linear.intercept_)

predictions  = linear.predict(x_test)

for x in range(len(predictions)):
    print (predictions[x], x_test[x], y_test[x])
   
p = 'studytime' 
style.use("ggplot")
plot.scatter(data[p],data["G3"])
plot.xlabel(p)
plot.ylabel('Final Grade G3')
plot.show()