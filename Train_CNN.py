# step 2 - Training the deep learning model and Storing the trained deep learning model
# Sequential is a class in Keras that allows you to create a 
# linear stack of layers for building neural network models.
from keras.models import Sequential
# Dense is a class that represents a fully connected (dense) layer in a neural network. 
# It's used for the hidden layers and the output layer of a neural network.
from keras.layers import Dense
# Flatten is a layer that's used to convert multidimensional arrays (such as images) into a 1D array, 
# which is required before feeding the data into fully connected layers.
from keras.layers import Flatten
# Conv2D is a class that represents a 2D convolutional layer. 
# Convolutional layers are used to extract features from images or other 2D data.
from keras.layers.convolutional import Conv2D
 #MaxPooling2D is a class for creating 2D max-pooling layers. 
# Max-pooling is used to downsample the spatial dimensions of the data, 
# reducing computational complexity and increasing the network's ability to learn invariant features.
from keras.layers.convolutional import MaxPooling2D
# to_categorical is a function that's used to convert categorical integer labels into one-hot encoded format.
# One-hot encoding is a common technique for representing categorical data as binary vectors, 
# where each class corresponds to a unique binary vector. 
# This is often used for the target labels in classification tasks.
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np

# Read the Data frame
preprocessed_data = pd.read_csv('train.csv', index_col=False)
# fetches 784th column
# The 784th column is the label column, which contains the digit that is represented by each image.
preprocessed_data.head()
# This line of code creates a new DataFrame labels by extracting the column labeled '784' 
# from the preprocessed_data DataFrame.
labels = preprocessed_data[['784']]
# Dropping the target variable from the dataframe to get only the features
preprocessed_data.drop(preprocessed_data.columns[[784]], axis=1, inplace=True)

# Convert labels series to numpy array
labels = np.array(labels)
# print(labels)

# Use the keras to_categorical function to apply one hot encoding
cat = to_categorical(labels, num_classes=14)
# print(cat.shape)

final = []
# Iterate over the number of rows in the data
for i in range(len(preprocessed_data)):
    # Reshape to 28x28 and append to a list
    final.append(np.array(preprocessed_data[i:i+1]).reshape(28, 28, 1))

# This line creates a new sequential Keras model, 
# which means that you'll be adding layers sequentially one after another.
model = Sequential()
# 16: The number of filters (also called kernels) in this convolutional layer.
# kernel_size=(5, 5): The size of the convolutional kernel. 
# A 5x5 kernel is used to extract features from the input image.
# input_shape=(28, 28, 1): The input shape of the layer. 
# In this case, it's a 28x28 grayscale image with a single channel.
# data_format='channels_last': The order of the dimensions in the input data. 
# 'channels_last' means that the channel dimension comes after the height and width dimensions.
# activation='relu': The activation function used for the output of this layer. 
# 'relu' stands for Rectified Linear Activation, which introduces non-linearity to the model.
model.add(Conv2D(16, kernel_size=(5, 5), input_shape=(28, 28, 1), data_format='channels_last', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# The Flatten layer is used to convert the 2D arrays resulting 
# from the convolutional and pooling layers into a 1D array.
# The rectified linear unit (ReLU) activation function helps introduce
# non-linearity to the model and is commonly used in hidden layers.
# Softmax converts the output values into probabilities, 
# ensuring that the sum of all output probabilities for each example is 1
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(14, activation='softmax'))
#Softmax -- converts ouput values to probability -- e^out / wum e^allpossout
model.summary()

# loss='categorical_crossentropy': This argument specifies the loss function that the model will use during training. 
# In this case, it's the categorical cross-entropy loss function.
# 'adam' refers to the Adam optimizer, which is an adaptive learning rate optimization 
# algorithm that works well for a wide range of problems.
# 'accuracy' is chosen as the metric. 
# The accuracy metric measures the proportion of correctly predicted labels to the total number of labels.
# categorical_crossentropy is a common loss function used for multi-class classification problems. 
# It calculates the cross-entropy loss between the true labels and the predicted probabilities.
# The Adam optimizer helps the model converge more quickly and efficiently during training.
# 'accuracy' metric, which calculates the ratio of correctly predicted instances to the total number of instances.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# np.array(final): This is the training data. final is a list of preprocessed images, and np.array() 
# converts the list into a NumPy array, which is the expected format for training data in Keras.
# The training dataset is divided into batches of size 200, and the model's weights are updated after processing each batch.
# verbose=1: This controls the verbosity of the training output
model.fit(np.array(final), cat, epochs=10, batch_size=200, shuffle=True, verbose=1)

# trained model to json file
#  This JSON representation includes information about the layers,
#  their types, activation functions, and other configuration details.
model_json = model.to_json()
with open("models/model_rev.json", "w") as json_file:
    json_file.write(model_json)
# The weights contain the learned parameters of the model after training. 
# This file can be used later to load the model with the same architecture and the learned weights.
model.save_weights("models/model_rev.h5")
