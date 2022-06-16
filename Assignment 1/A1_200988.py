#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      asus
#
# Created:     09-06-2022
# Copyright:   (c) asus 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression:

    # The __init__ is called when we make any object of our class. Here, you are to specify the default values for
    # Learning Rate, Number of Iterations, Weights and Biases. It doesn't return anything.
    # Hint: Google what a `self pointer` is and figure out how it can be used here.


    def __init__(self, learning_rate=0.001, n_iters=1000):
         self.learning_rate=learning_rate
         self.n_iters=n_iters
         self.w=None
         self.b=None

    # The following function would be the heart of the model. This is where the training would happen.
    # You're supposed to iterate and keep on updating the weights and biases according to the steps of Gradient Descent.
    def fit(self, X, y):
        row,col=X.shape
        self.w=np.zeros(col)
        self.b=0
        for i in range(self.n_iters):
            y_pred=np.dot(X,self.w)+self.b

            dw=(2/(row*col))*(np.dot(X.transpose(),y_pred - y))
            db=(2/(row*col))*(np.sum(y_pred - y))

            self.w=self.w-self.learning_rate*dw
            self.b=self.b-self.learning_rate*db
    # This function will be called after our model has been trained and we are predicting on unseen data
    # What is our prediction? Just return that
    def predict(self, X):

            y_pred=np.dot(X,self.w)+self.b
            return y_pred
# Generate the data
X, y = datasets.make_regression(n_samples=100, n_features=5, noise=20, random_state=4)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Now, we make an object of our custom class.
regressor = LinearRegression(0.0118) # You may pass the custom parameters or let the default values take it ahead

# Call the fit method on the object to train (pass appropriate part of dataset)
regressor.fit(X_train,y_train)
# Now, let's see our what our model predicts
predictions = regressor.predict(X_test) # pass appropriate part of dataset

def mean_squared_error(y_true, y_pred):
       # return the mean squared error
       return np.mean((y_true-y_pred)**2)

def r2_score(y_true, y_pred):
      # return the r2 score
       return (np.corrcoef(y_true,y_pred))**2

mse = mean_squared_error(y_test,predictions) # Pass appropriate parts of dataset
print("MSE:", mse)
accu = r2_score(y_test,predictions) # Pass appropriate parts of dataset
print("Accuracy:", accu[0][1])

Y=regressor.predict(X)
plt.plot(X,Y, 'o r')
plt.show()