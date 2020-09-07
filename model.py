import pandas
import numpy as np
import cv2

#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from math import floor

from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Conv2D, ELU, Flatten, Dense
from keras.optimizers import Adam

import random

# globals
columns = 200
rows = 66
channels = 3
crop_top = 50
crop_bottom = 25

def convert_grayscale(image, steering):
    gray = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)
    # convert it back to rgb
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray, steering

def translate_image(image, steering):
    range = 10 #max 10px movement
    trans_x = range*np.random.uniform()-range/2
    trans_y = range*np.random.uniform()-range/2
    TranslationMatrix = np.float32([[1,0,trans_x],[0,1,trans_y]])
    
    translated = cv2.warpAffine(image,TranslationMatrix,(image.shape[1],image.shape[0]))
    
    # recalculate steering angle for horizontal 
    steering += trans_x * 3.75e-4
    return translated, steering

def flip_image(image, steering):
    image = cv2.flip(np.copy(image),1)
    steering = -steering
    return image, steering

# convert first to hsv and then apply the value
def add_brightness(image, steering):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = 0.5 + 0.4 * (2 * np.random.uniform()  - 1.0)
    cv2.add(hsv[:,:,2], value, hsv[:,:,2])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image, steering

def no_augmentation(image, steering):
    return image, steering

augmentation = [no_augmentation, convert_grayscale, translate_image, flip_image, add_brightness]

def get_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # remove sky and hood of the car
    image = image[35:135,:,:] # 35px from top 25 px from bottom
    
    # resize to nvidias neural network
    image = cv2.resize(image, (columns, rows), interpolation = cv2.INTER_AREA)
    
    return np.array(image)


def training_samples_generator(paths, steering_angles, batch_size):
    images_batch = np.zeros((batch_size, rows, columns, channels))
    steering_angles_batch = np.zeros(batch_size)
    
    while 1:
        for i in range(batch_size):
            index = np.random.randint(len(paths))
            image = get_image(paths[index])
            
            steering = steering_angles[index]
            
            augment = np.random.choice(augmentation)
            image, steering = augment(image, steering)
            
            images_batch[i] = image
            steering_angles_batch[i] = steering
        
        yield images_batch, steering_angles_batch

def validation_samples_generator(paths, steering_angles, batch_size):
    images_batch = np.zeros((batch_size, rows, columns, channels))
    steering_angles_batch = np.zeros(batch_size)
    
    while 1:
        for i in range(batch_size):
            index = np.random.randint(len(paths))
            image = get_image(paths[index])
            
            steering = steering_angles[index]
            
            
            images_batch[i] = image
            steering_angles_batch[i] = steering
        
        yield images_batch, steering_angles_batch
    
            
# create model using nvidia
def create_model():
     model = Sequential()
     #model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(66,200,channels))) # remove horizon + car hood
     #model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # normalize data
     model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(rows,columns,3)))   
     model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))
     model.add(ELU())
     model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))
     model.add(ELU())
     model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))
        
     model.add(ELU())
     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer="he_normal"))
     model.add(ELU())
     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer="he_normal"))
     model.add(ELU())
     #model.add(Dropout(0.80))
     model.add(Flatten())
     model.add(Dense(1164, kernel_initializer='he_normal'))
     model.add(ELU())
     model.add(Dense(100, kernel_initializer='he_normal'))
     model.add(ELU())
     #model.add(Dropout(0.80))
     model.add(Dense(50, kernel_initializer='he_normal'))
     model.add(ELU())
     model.add(Dense(10, kernel_initializer='he_normal'))
     model.add(ELU())
     model.add(Dense(1, name='output', kernel_initializer='he_normal'))
    
     model.compile(loss='mean_squared_error', optimizer='adam')
     #model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')   
        
     return model

        
# load data
#path = "/opt/testdata/reentering/"
#path = "/opt/testdata/normal_lap/"
path = "data/"
csv_file = path + "driving_log.csv"
#center,left,right,steering,throttle,brake,speed
data = pandas.read_csv(csv_file, header= 0, names = ['center', 'left', 'right', 'steering', 'throttle','brake','speed'])

image_paths = []
steering_angles = []

for row in range(int(len(data))):
    row_data = data.iloc[[row]].reset_index()

    image_paths.append(path+row_data['center'][0].strip())
    image_paths.append(path+row_data['left'][0].strip())
    image_paths.append(path+row_data['right'][0].strip())
    steering_angles.append(row_data['steering'][0])
    steering_angles.append(row_data['steering'][0] + 0.25)
    steering_angles.append(row_data['steering'][0] - 0.25)
    
    image_paths.append(path+row_data['center'][0].strip())
    image_paths.append(path+row_data['left'][0].strip())
    image_paths.append(path+row_data['right'][0].strip())
    steering_angles.append(row_data['steering'][0])
    steering_angles.append(row_data['steering'][0] + 0.25)
    steering_angles.append(row_data['steering'][0] - 0.25)
   

image_paths = np.array(image_paths)
steering_angles = np.array(steering_angles)

#quit() 

# split the data into test and training 80% training 20% test
paths_train, paths_validation, steering_train, steering_validation = train_test_split(image_paths, steering_angles, test_size = 0.2)

print ("Training Data: ", len(paths_train))
print ("Test Data: ", len(paths_validation))

batch_size = 64

validation_generator = validation_samples_generator(paths_validation, steering_validation, batch_size)
training_generator = training_samples_generator(paths_train, steering_train, batch_size)

training_batches = int(floor(len(paths_train) / batch_size))
validation_batches = int(floor(len(paths_validation) / batch_size))

print("Total validation batches: ", int(floor(len(paths_validation) / batch_size)))
print("Total training batches: ", int(floor(len(paths_train) / batch_size)))

model = create_model()
history = model.fit_generator(training_generator, steps_per_epoch=training_batches, validation_data=validation_generator, validation_steps=validation_batches, epochs=4, verbose = 1)
model.save("model.h5")