#************************************************
# Logistic regression tutorial
# Purpose: to plot and classify data with labels and provide a boundary line whose error will be reduced. In terms of a perceptron, this linear model, with some equation, can identify unlabeled inputs and output the probaility of being in the positive region (this feedforward--turning inputs into outputs).
#************************************************
import numpy as np
import matplotlib.pyplot as plt

#================================================
# Draw the regression line for classification
#================================================
def draw(x1, x2):
    ln = plt.plot(x1, x2)
    plt.pause(0.0001) # in secs
    ln[0].remove() # remove old line with new to make a cool animation

#================================================
# Compute and return the  result of sigmoid equation
#================================================
def sigmoid(score):
    return 1/(1+np.exp(-score))

#================================================
# Calculate probability of every point being positive
#================================================
def calculate_error(line_parameters, points, y):
    m = points.shape[0] # number of points = number of rows
    p = sigmoid(points*line_parameters) # plug all points into the sigmoid equation
    cross_entropy = -(1/m) * (np.log(p).transpose() * y + np.log(1-p).transpose() * (1-y))
    return cross_entropy

#================================================
# Calculate new gradient descent
# Note: alpha is a learning-rate variable
# At every iteration we minimize the error of some line by subtracting the gradient of the error. This creates a new line with a smaller error, and keep doing that 500 times.
#================================================
def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    
    for i in range(1000):
        p = sigmoid(points*line_parameters) # plug all points into the sigmoid equation
        gradient = (alpha/m) * points.transpose() * (p - y)
        
        # Keep updating weights and bias
        line_parameters = line_parameters - gradient
        
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()]) # (left-most pt, right-most pt)
        # w1x1 + w2x2 + b = 0, thus
        x2 = -b/w2 + x1 * (-w1/w2) # create x2 (vertical axis) values based on both x1 values
        draw(x1, x2)
        
        # Calculate error for current regression line
        print(calculate_error(line_parameters, points, y))

#------------------------------------------------
# Declarations
n_pts = 100 # set number of points to create
np.random.seed(0) # seed random number generator
bias = np.ones(n_pts) # bias is the third node in perceptron, but it's just 1

#------------------------------------------------
# Create values (x1 and x2 components) for the TOP region using normal distribution (multiplier, st dev, # pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).transpose()
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).transpose()

# Vertically stack arrays, add bottom below top (as a set, thus one argument)
all_pts = np.vstack((top_region, bottom_region))

# Prepare matrix to calculate outputs for the linear combination (start with all parameters being zero)
line_parameters = np.matrix([np.zeros(3)]).transpose() # transpose to multiply with the top and bottom regions

y = np.array([(np.zeros(n_pts), np.ones(n_pts))]).reshape(n_pts * 2, 1) # n points for zeros and n points for ones

#------------------------------------------------
# Label points and show plot
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], color = 'r') # label red
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color = 'b') # label blue
gradient_descent(line_parameters, all_pts, y, 0.06)
plt.show()
