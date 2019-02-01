 #************************************************************************
 # MNIST Image Recognition Tutorial
 #************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential # allows us to define our neural model
from keras.layers import Dense # allows us to connect preceeding layers in the network to proceeding creating a fully connected layered network.
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical # to be able to deal with multiclass datasets
import random
import requests # send a request over the internet
from PIL import Image # Python Imaging Library
import cv2 # to be able to resize the test image from the url

# Seed random num generator
np.random.seed(0)
# Store training data, test data (with labels--y)
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("Initial training data layout:\n", x_train.shape)
print("Initial test data layout:\n", x_test.shape)
print("Amount of labeled data:\n", y_train.shape[0])


# Take in an argument, true or false, if condition is met, the program will go on, else an error message will alert the user. This function call is very good practice in more complex networks as it helps with debugging.
assert(x_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels (training data)."
assert(x_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels (test data)."
assert(x_train.shape[1:] == (28, 28)), "The dimensions of the images are not 28x28."
assert(x_test.shape[1:] == (28, 28)), "The dimensions of the images are not 28x28."

# Record images in each of the 10 categories, 5 images of each class, and display it on a grid
num_of_samples = []

cols = 5
num_classes = 10
'''
fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5,10))
fig.tight_layout()

# Display "random" numbers starting with 0 at the top. Display in greyscale.
for i in range(cols):
    for j in range(num_classes):
        x_selected = x_train[y_train == j] # get image
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected-1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis('off')
        # Label middle column of rows
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))
'''
'''
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title('Distribution of the training dataset')
plt.xlabel('class number')
plt.ylabel('number of images')
'''

# Prepare data for training neural network
# Hot encode data label
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Normalize values, divide by 255 to reduce pixel intensities to be between 0 and 1. Since we have greyscaled it before. This makes it easier for the network to learn.
x_train = x_train/255
x_test = x_test/255

# Flatten our 28x28 2d image into a 784-long array
num_pixels = 784
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)
#print("New layout of training data:\n", x_train.shape)
#print("New layout of test data:\n", x_test.shape)

#=========================================================================
# Create a neural network
'''
Using activation function ReLU (Rectifier Linear Unit).
* Softmax function: converts all scores to probabilities.
'''
#=========================================================================
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(10, activation='relu'))
    '''
    Had to play with the hidden layers in order to get the test digit correctly classified.
    '''
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) # output layer
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#-------------------------------------------------------------------------
model = create_model()
print(model.summary())
'''
Note: a deep neural network is not scalable when dealing with larger images.
'''
# 10 % of training data set aside for validation
history = model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=200, verbose=1, shuffle=1)

'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('epoch')
'''

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

# Predict the class of a handwritten image, the digit 2.
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True) # Take image from url
img = Image.open(response.raw)
# Convert input data into an array
img_array = np.asarray(img)
resized = cv2.resize(img_array, (28, 28)) # resize to 28x28
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # reduce color

image = cv2.bitwise_not(gray_scale) # invert intensities channel to one (gray)
plt.imshow(image, cmap=plt.get_cmap('gray'))

image = image/255 # normalize intensities
image = image.reshape(1, 784)

# Make a prediction
prediction = model.predict_classes(image)
print('predicted digit: ', str(prediction))

'''
Note: In order to get more accuracy, we will make use of a convoluted neural network.
'''

plt.show()
