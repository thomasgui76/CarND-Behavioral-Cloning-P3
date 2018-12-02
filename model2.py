import tensorflow as tf
import keras
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense,Dropout,Lambda
from keras.layers import Conv2D, MaxPooling2D,Cropping2D
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import csv
import numpy as np
from keras.utils import plot_model
from matplotlib import pyplot as plt

# read data record from driving_log.csv
# load 3 images from center,left,right camera
# adjust steering angles for left, right camera images
image_files = []
angles = []
with open('./track1/driving_log.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.3 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        path = './track1/IMG/'
        image_files.append(path+line[0].split('/')[-1])
        image_files.append(path+line[1].split('/')[-1])
        image_files.append(path+line[2].split('/')[-1])
        angles.append(steering_center)
        angles.append(steering_left)
        angles.append(steering_right)

train_image_files, validation_image_files,train_angles, validation_angles = train_test_split(image_files,
        angles,test_size=0.2)

print("number of images in training set is {}".format(len(train_image_files)))
print("number of angles in training set is {}".format(len(train_angles)))
print("number of images in validation set is {}".format(len(validation_image_files)))
print("number of angles in validation set is {}".format(len(validation_angles)))


# define data generator
def generator(X_image_files, y_angles, batch_size):
    sample_nums = len(y_angles)
    # print('sample length: ',sample_nums)
    while 1:  # Loop forever so the generator never terminates
        shuffle(X_image_files, y_angles)
        for offset in range(0, sample_nums, batch_size):
            batch_image_files = X_image_files[offset:offset + batch_size]
            # print('batch_images length: ',len(batch_images))
            batch_angles = y_angles[offset:offset + batch_size]
            # print('batch_angles length: ',len(batch_angles))

            images = []
            angles = []
            for image_file, angle in zip(batch_image_files, batch_angles):
                image = cv2.imread(image_file)
                # image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


batch_size = 32
train_generator = generator(train_image_files, train_angles, batch_size)
validation_generator = generator(validation_image_files, validation_angles, batch_size)

# define model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# trip image, output shape (66,320,3)
model.add(Cropping2D(cropping=((70,25), (20,20))))
# conv layer1, kernel size=5x5,output channel=24, stride=(2,2)
# output shape=(31,158,24)
model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
# conv layer2, kernel size=5x5, output channel=36, stride=(2,2)
# output shape=(14,77,36)
model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
# conv layer3, kernel size=5x5, output channel=48, stride=(2,2)
# output shape=(5,37,48)
model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
# conv layer4, kernel size=3x3,output channel=64,stride=(1,1)
# output shape=(3,35,64)
model.add(Conv2D(64,(3,3),activation='relu'))
# conv layer5, kernel size=3x3,output channel=64,stride=(1,1)
# output shape=(1,33,64)
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
# plot_model(model, to_file='./examples/model2.png')

model.compile(loss='mse',optimizer='adam')
model_history=model.fit_generator(train_generator, steps_per_epoch= \
            len(train_angles)//batch_size, validation_data=validation_generator, \
            validation_steps=len(validation_angles)//batch_size, epochs=7)
model.save('./model2.h5')
model.reset_states()

### plot the training and validation loss for each epoch
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
xticks = np.arange(0,len(model_history.history['loss']),1)
plt.xticks(xticks)
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()