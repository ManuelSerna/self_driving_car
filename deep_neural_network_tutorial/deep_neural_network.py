#*************************************************************************
# Deep Neural Network tutorial
#*************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import datasets # get more complicated data sets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#=========================================================================
# Coorelate each specific coordinate with its respective prediction probability.
'''
X: refers to data created (Xa and Xb)
y: matrix that contains labels for our data set (1 for bottom, 0 for top)
model: model that was trained to fir our data 
'''
#=========================================================================
def plot_decision_boundary(X, y, model):
    # Find leftmost point (first arg), and rightmost (second arg) to make an evenly spaced graph
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25, 50) # horizontal components in column 0
    # Find topmost and bottom-most points
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25, 50) # vertical components in column 0
    
    # Take each argument matrix and convert it to a 50x50. xx has repeated rows, yy has repeated columns.
    xx, yy = np.meshgrid(x_span, y_span)
    
    xx_, yy_ = xx.ravel(), yy.ravel() # make arrays one dimensional
    grid = np.c_[xx_, yy_]
    
    # Trained model tests all the points and makes predictions. It will decide if points are to be labeled 0 or 1.
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z) # display contour zones

#-------------------------------------------------------------------------
np.random.seed(0)
n_pts = 500
# Store values in X, and labels in y
'''
Positive region: 1, circle at center.
Negative region: 0, ring around circle.

- Random state allows us to seed our random number generator, so we can reproduce the same random numbers.
- Noise ensures data has some variance, so neural network is being challenged.
    - A high noise (like 0.8) will cause the data points to become overly convoluted, making it a lot harder to classify the data, you want to ensure that the noise is kept relatively low (like 0.1).
- Factor, positive region will be smaller than negative region.
'''
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

#print(X)
#print(y)
#plt.scatter(X[y==0, 0], X[y==0, 1]) # get all (x,y) coords whose label corresponds to a label of 0
#plt.scatter(X[y==1, 0], X[y==1, 1]) # now get all (x,y) coords whose label corresponds to a label of 1

model = Sequential() # define neural network as a sequential model
'''
Add input layer and first hidden layer
- Add 4 inner layer neurons (it is best to use 4 inner layer nodes to best fit our inner circle.
- Input shape: number of input nodes, only dealing with x1 and x2, so 2.
- Use sigmoid activation function.
'''
model.add(Dense(4, input_shape=(2, ), activation='sigmoid'))
'''
Add the final (the ouput) layer.
Thus, we have two input nodes that are connected to 4 nodes in the hidden which connect to 1 node in the final layer.
    - Input layer with 2 nodes.
    - One hidden layer with 4 nodes.
    - Output layer with 1 node.
'''
model.add(Dense(1, activation='sigmoid'))

# We must now compile our model. Use the Adam optimizer. Only differentiate between two classifications, so use binary cross entropy.
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy'])

# Train the model to fit our data.
h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle='true')
'''
plt.plot(h.history['acc'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title(['accuracy'])
'''
'''
plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend(['loss'])
plt.title(['loss'])
'''
'''
# Get probability regions
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1]) # plot as scatter
plt.scatter(X[n_pts:,0], X[n_pts:,1])
'''
# Make a prediction given x,y values
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1]) # plot as scatter
plt.scatter(X[n_pts:,0], X[n_pts:,1])

x = 0.1
y = 0
point = np.array([[x, y]])

prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='red')
print("prediction is: ", prediction) # probability that point is in region 1
'''
Deep Neural Networks, when faced with large amounts of data, use up a lot of computing resources. Convoluted neural networks are a much more effecient solution.
'''

plt.show()
