import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing import image


def preprocess_pixels(pixel_data):
  images = []
  for i in range(len(pixel_data)):
    img = np.fromstring(pixel_data[i], dtype='int', sep=' ')
    img = img.reshape(48,48,1)
    images.append(img)
  X = np.array(images)
  return X


def emotion_recognition(input_shape):
  X_input = Input(input_shape)
  #layer1
  X = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid')(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  #layer2
  X = Conv2D(64, (3,3), strides=(1,1), padding = 'same')(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  #layer3
  X = Conv2D(64, (3,3), strides=(1,1), padding = 'valid')(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  #layer4
  X = Conv2D(128, (3,3), strides=(1,1), padding = 'same')(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  #layer5
  X = Conv2D(128, (3,3), strides=(1,1), padding = 'valid')(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  X = Flatten()(X)
  
  X = Dense(200, activation='relu')(X)
  X = Dropout(0.6)(X)
  X = Dense(7, activation = 'softmax')(X)

  model = Model(inputs=X_input, outputs=X)

  return model


def get_class(preds):
  pred_class = np.zeros((preds.shape[0],1))
  for i in range(len(preds)):
   pred_class[i] = np.argmax(preds[i])

  return pred_class


label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
#read dataset
data = pd.read_csv('fer2013.csv')

#separate pixels and emotions
pixel_data = data['pixels']
label_data = data['emotion']

#randomly duplicates minority classes reducing the imbalance in the dataset
oversampler = RandomOverSampler(sampling_strategy='auto')
X_over, Y_over = oversampler.fit_resample(pixel_data.values.reshape(-1,1), label_data)
X_over_series = pd.Series(X_over.flatten())

#pixel to image form
X = preprocess_pixels(X_over_series)
#reshape emotion
Y = Y_over.values
Y = Y.reshape(Y.shape[0],1)

#split testset and trainset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 45)

y_train = to_categorical(Y_train, num_classes=7)
y_test = to_categorical(Y_test, num_classes=7)
#print(f"X_train :{len(X_train)}")
#print(f"X_test :{len(X_test)}")
#plt.imshow(X[53,:,:,0])

#Create model
model = emotion_recognition((48,48,1))
model.summary()
adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#Train model
model.fit(X_train, y_train,epochs=35,validation_data=(X_test, y_test))

fer_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("model2.h5")


'''
#Validation
preds = model.predict(X_train)
pred_class_train = get_class(preds)'''

'''img_path = 'test1_happiness.jpeg'
img = image.load_img(img_path, grayscale=True, target_size=(48,48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

prediction = np.argmax(model.predict(x))
print('The predicted emotion is : ' + label_dict[prediction])
my_image = image.load_img(img_path)
plt.imshow(my_image)'''
