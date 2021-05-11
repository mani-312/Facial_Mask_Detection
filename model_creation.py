print("[INFO] Importing modules")
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2
import keras

def load_images(folder,data,labels,label):
    i=0
    for file in os.listdir(folder):
        imagePath = os.path.join(folder,file);
        im = load_img(imagePath,target_size=(224,224)) #Reads in RGB format
        im = img_to_array(im)
        im = preprocess_input(im)
        data.append(im)
        labels.append(label)
        print(i)
        i=i+1
    return data,labels

print("[INFO] Loading images")
data=[]
labels=[]
data,labels=load_images("C:/Users/MANIKANTA/Desktop/covid/face-mask-detector/dataset/with_mask",data,labels,1)
data,labels=load_images("C:/Users/MANIKANTA/Desktop/covid/face-mask-detector/dataset/without_mask",data,labels,0)
data = np.array(data,dtype="float32")
labels = np.array(labels)

# defining Number of epochs to train and batch size
EPOCHS = 20
BS = 32

# Splitting the data into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=0)

input_shape = x_train.shape[1:]

y_train =  utils.to_categorical(y_train.reshape(y_train.shape[0],1))
y_test =  utils.to_categorical(y_test.reshape(y_test.shape[0],1))

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

# defining the architecture of CNN model
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(input_shape)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128,activation="relu"))
model.add(tf.keras.layers.Dense(units=32,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=32,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=2,activation="softmax"))

# Training model using sgd algorithm
model.compile(loss='categorical_crossentropy',
optimizer='sgd',
metrics=['accuracy'])

print("[INFO] Training model")
history = model.fit(
	aug.flow(x_train, y_train, batch_size=BS),
	steps_per_epoch=len(x_train) // BS,
	validation_data=(x_test, y_test),
	validation_steps=len(x_test) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(x_test, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(y_test.argmax(axis=1), predIdxs))


# Plotting the loss and accuracy of the model over each epoch
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Augmented_CNN_performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,11))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.show()
