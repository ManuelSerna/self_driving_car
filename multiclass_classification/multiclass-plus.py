#*************************************************************************
# Multiclass classification tutorial
'''
This program will categorically classify MULTIPLE labels
Builds off of the deep neural network and perceptron tutorials.
Uses the keras library.
'''
#*************************************************************************
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense # connect preceeding layers to proceeding layers in the neural networks
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

#=========================================================================
# Coorelate each specific coordinate with its respective prediction probability.
'''
X: refers to data created (Xa and Xb)
y: matrix that contains labels for our data set (1 for bottom, 0 for top)
model: model that was trained to fir our data 
'''
#=========================================================================
def plot_decision_boundary(X, y_cat, model):
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 50)
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict_classes(grid) # predict for multiclass
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

#-------------------------------------------------------------------------

n_pts = 500
centers = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]] # "focii" of the 5 data blobs
'''
Arg 1: define number of samples/points
Arg 2: seed random number generator
Arg 3: give centers for each category
Arg 4: the distance between each data point
'''
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4) # make blobs in different coords in the grid

# Scatter plot for data
'''
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.scatter(X[y==3, 0], X[y==3, 1])
plt.scatter(X[y==4, 0], X[y==4, 1])
#'''

# Replace numerical labels with hot encoded labels
print(y)
y_cat = to_categorical(y, 5) # labels, number of data classes (in our case 3)
print(y_cat)

# Define neural network, one with an input layer, and an output layer
model = Sequential()
model.add(Dense(units=5, input_shape=(2, ), activation='softmax')) # number of ouput nodes (the next layer), input nodes, using the softmax function.
model.compile(Adam(0.1), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X, y=y_cat, verbose=1, batch_size=50, epochs=100) # train the model (with hot-encoded labels)


# Plot final model
plot_decision_boundary(X, y_cat, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.scatter(X[y==3, 0], X[y==3, 1])
plt.scatter(X[y==4, 0], X[y==4, 1])
#'''

# Predict probability of a value using the trained model
'''
plot_decision_boundary(X, y_cat, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.scatter(X[y==3, 0], X[y==3, 1])
plt.scatter(X[y==4, 0], X[y==4, 1])
x = 0.5
y = 0.5
point = np.array([[x, y]])
prediction = model.predict_classes(point)
plt.plot([x], [y], marker='o', markersize=10, color='r')
print('Prediction is: ', prediction)
#'''

plt.show()
