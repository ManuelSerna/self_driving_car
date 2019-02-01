# -*- coding: utf-8 -*-
"""traffic_signs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jFn5LATobwo9CYq3EIjL1_okVNviGwXl
"""

!git clone https://bitbucket.org/jadslim/german-traffic-signs

!ls german-traffic-signs

#******************************************************************
'''
Classifying Road Symbols
* Note: This project was originally created on a python notebook (Google Colaboratory). This was done to take advantage of the free GPU power offered, and getting data from the repository service bitbucket.
- Need to preprocess images first before we can pass them as inputs in the cnn
'''
#******************************************************************
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle # pickle serializes images
# manipulate and analyze data inside of a csv file (comma-separated-files)
import pandas as pd
import random

np.random.seed(0)

#******************************************************************
# Preprocess images in several steps
#******************************************************************

# Unpickle contents of f into each respective data set
with open('german-traffic-signs/train.p', 'rb') as f:
  train_data = pickle.load(f) 
with open('german-traffic-signs/valid.p', 'rb') as f:
  val_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
  test_data = pickle.load(f)
  
print(type(train_data))
X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# Visualize size and dimensions (32x32) of data sets, the traffic signs have 3 color channels (RGB). At this point we have effectively imported our data.
print('training data: ', X_train.shape)
print('validation data: ', X_val.shape)
print('testing data: ', X_test.shape)

# Assert the number of images equals the number of labels
assert(X_train.shape[0] == y_train.shape[0]), 'The number of images is not equal to the number of labels'
assert(X_val.shape[0] == y_val.shape[0]), 'The number of images is not equal to the number of labels'
assert(X_test.shape[0] == y_test.shape[0]), 'The number of images is not equal to the number of labels'
# Assert that the training image at index 1 and beyond is 32x32x3
assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32 x 3"
assert(X_val.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32 x 3"
assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32 x 3"

# Load from csv file
data = pd.read_csv('german-traffic-signs/signnames.csv')
#print(data)

# Display selected images (randomly out their set) in a grid (without axis labels)
num_of_samples=[]
 
cols = 5
num_classes = 43
 
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
fig.tight_layout()

# (index, Series)
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + '-' + row["SignName"])
            num_of_samples.append(len(x_selected))

'''
Plot number of samples belonging to each class.
- As you will see, some classes have MUCH more images to train with than others.
  (which is natural, some signs appear more than others)

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()
'''

# We will need to reduce the color channels to one (greyscale) so that our neural network can process the images with less difficulty
import cv2

'''
Steps to preprocess an image (example in the next following lines).
    1 Grayscale.
    2 Equalize grayscale intensities.
    3 Normalize grayscale intensities (divide by 255).
plt.imshow(X_train[1000])
plt.axis('off')
print(X_train[1000].shape)
print(y_train[1000])
'''

#==================================================================
# Turn RGB images into greyscale
'''
Reinforce that color is not a very defining feature for these signs. Greyscaling them is enforcing this.
Also, reduce the depth from 3 to 1, fewer parameters are needed, 
so the neural network is much more efficient and require less computing power to classify data.
'''
#==================================================================
def grayscale(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey
#------------------------------------------------------------------
'''
# Grayscale an example image (and display it)
img = grayscale(X_train[1000])
plt.imshow(img)
plt.axis('off')
print(img.shape)
'''

#==================================================================
# Equalize grayscale intensities
'''
Histogram equalization: aims to standardize the lighting in all images.
This enhances the contrast such that any grayscale intensities are better distributed.
'''
#==================================================================
def equalize(img):
    equal = cv2.equalizeHist(img) # function will only accept grayscale images
    return equal
#------------------------------------------------------------------
'''
# Example of equalizing on one image (and displaying)
img = equalize(img)
plt.imshow(img) # Notice the more defined features in the equalized image
plt.axis('off')
print(img.shape)
'''

#==================================================================
# Preprocess entire data set
#==================================================================
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255 # normalize gray intensities
    return img # return preprocessed image
#------------------------------------------------------------------
# Preprocess all data sets
X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

'''
# Example of calling preprocessing function (and displaying)
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
print(X_train.shape)
'''

# Format arrays to be 1D so the cnn can process them
X_train = X_train.reshape(34799, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)

'''
# Print shape to confirm arrays are now 1D
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
'''

# One hot encode data labels
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
y_test = to_categorical(y_test, 43)

# Now we have properly preprocessed our images.
#******************************************************************

#==================================================================
# Utilize leNet convolutional neural network
#==================================================================
def leNet_model():
    model = Sequential()
    # Add layers, start with convolutional layer with 30 5x5 filters, we will not modify the stride and padding args
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    # Add pooling layer (scale down feature maps into a small, generalized representation, helps to avoid overfitting)
    model.add(MaxPooling2D(pool_size=(2, 2))) # (2,2), so scale down to half the size, same depth of 30
    # Add another convoltional layer
    model.add(Conv2D(15, (3, 3), activation='relu'))
    # Feed input into second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten array of inputs in order to format properly so that it can be fed into the fully-connected layer
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5)) # drop half of all input nodes at each update
    # Define output layer
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#------------------------------------------------------------------

model = leNet_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=430, verbose=1, shuffle=1)

'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
'''
'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
'''
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

'''
- We now have a quantitative understanding of our network's performance and it's not very good.
- Looking at the loss, it is even higher at a minimum value of about ~0.9.
- Validation accuracy seems to lag behind of training accuracy, which is at about 96%.

These values imply that our network is not performing effectively in terms of accurately predicting images from the dataset.
    - What's more our network seems to have overfit in our data as well becuase the validation acc trails behind the training acc.
    - We must finetune our model to improve performance.

* 2 Main issues:
    - Accuracy is not as high as we would like
    - Network seems to overfit the training data.

Always try to modify your model to see just how these modifications improve the effectiveness of your model.
'''
