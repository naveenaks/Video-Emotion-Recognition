import sys
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


df=pd.read_csv('fer2013.csv')

#print(df.info())
#print(df["Usage"].value_counts())
#print(df.head())


X_train,train_y,X_test,test_y=[],[],[],[]

num_features = 64
num_labels = 7
batch_size = 64
epochs = 1
width,height = 48, 48

#x Axis - rows contain pixels and y-axis contain the emotion.
for index,row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if "Training" in row['Usage']:
            X_train.append(np.array(val,'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")
        
#print(f"X_train.sample data :{X_train[0:4]}")
#print(f"train_y.sample data :{train_y[0:4]}")

#print(f"X_train.sample data :{X_train[0:4]}")
#print(f"train_y.sample data :{train_y[0:4]}")
#print(f"X_test.sample data :{X_test[0:4]}")
#print(f"test_y.sample data :{test_y[0:4]}")


#Converting into proper matrices
X_train=np.array(X_train,'float32')
train_y=np.array(train_y,'float32')
X_test=np.array(X_test,'float32')
test_y=np.array(test_y,'float32')

#print(f"X_train.sample data :{X_train[0:4]}")
#print(f"train_y.sample data :{train_y[0:4]}")


#Normalising data btw 0 and 1
#Subtract mean then divide by deviation of every row (mean) hence, axis =0.
X_train-=np.mean(X_train,axis=0)
X_train/=np.std(X_train,axis=0)
X_test-=np.mean(X_test,axis=0)
X_test/=np.std(X_test,axis=0)

#print(f"X_train.sample data :{X_train[0:4]}")
#print(f"train_y.sample data :{train_y[0:4]}")

X_train = X_train.reshape(X_train.shape[0],width,height,1)
X_test = X_test.reshape(X_test.shape[0],width,height,1)
'''
print(f"X_train.sample data :{X_train[0:1]}")
print(f"train_y.sample data :{train_y[0:1]}")
'''



train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)


#designing cnn in keras
#1st layer
#
model = Sequential()


model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

#converted successfully into a 1D network
model.add(Flatten())

#add the dense layers
model.add(Dense(2*2*2*2*num_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2*2*2*2*num_features,activation='relu'))
model.add(Dropout(0.2))

#multiclass classification
model.add(Dense(num_labels, activation='softmax'))



model.compile(loss=categorical_crossentropy,optimizer=Adam(),metrics=['accuracy'])
model.fit(X_train, train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, test_y),shuffle=True)


#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
