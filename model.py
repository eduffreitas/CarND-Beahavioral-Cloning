import csv
import numpy as np
import cv2
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
import os.path
import matplotlib.pyplot as plt
np.random.seed(42)



if not(os.path.exists('train.csv') and os.path.exists('validation.csv')):

    path_to_replace = "C:\\Users\\eduardo\\Documents\\SelfDrivingCar\\beta-simulator-windows\\beta_simulator_windows\\data"

    def ReplaceWrongPath(value):
        return value.replace(path_to_replace, "").replace("\\", "/").replace(" ", "")

    X_train = []
    y_train = []
    X_train_left = []
    y_train_left = []
    X_train_right = []
    y_train_right = []

    with open('./data/driving_log.csv', 'r') as csv_file_in:

        csv_reader = csv.DictReader(csv_file_in)

        for row in csv_reader:
            steering = float(row['steering'])

            #center image
            path = './data/' + ReplaceWrongPath(row['center'].strip())        
            X_train.append(path)
            y_train.append(steering)
            continue
            
            if steering == 0:
                continue
                
            if steering < 0:
                #left image
                path = './data/' + ReplaceWrongPath(row['left'].strip())
                steering_left = steering + 0.2

                X_train_left.append(path)
                y_train_left.append(steering_left)

                #right image
                path = './data/' + ReplaceWrongPath(row['right'].strip())
                steering_right = steering - 0.2
                steering_right = steering_right if steering_right > -1 else -1

                X_train_right.append(path)
                y_train_right.append(steering_right)
            else:
                #left image
                path = './data/' + ReplaceWrongPath(row['left'].strip())
                steering_left = steering + 0.2
                steering_left = steering_left if steering_left < 1 else 1

                X_train_left.append(path)
                y_train_left.append(steering_left)

                #right image
                path = './data/' + ReplaceWrongPath(row['right'].strip())
                steering_right = steering - 0.2

                X_train_right.append(path)
                y_train_right.append(steering_right)

    X_train, y_train = shuffle(X_train, y_train)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    [X_train.append(item) for item in X_train_left]
    [X_train.append(item) for item in X_train_right]
    [y_train.append(item) for item in y_train_left]
    [y_train.append(item) for item in y_train_right]
    
    X_train, y_train = shuffle(X_train, y_train)

    with open('train.csv', 'w') as csv_file_train:

        fieldnames = ['path','steering']
        writer = csv.DictWriter(csv_file_train, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(X_train)):
            path, steering = X_train[i], y_train[i]
            writer.writerow({'path': path, 'steering': steering})

    with open('validation.csv', 'w') as csv_file_train:

        fieldnames = ['path','steering']
        writer = csv.DictWriter(csv_file_train, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(X_validation)):
            path, steering = X_validation[i], y_validation[i]
            writer.writerow({'path': path, 'steering': steering})

    print("processing done")
else:
    print("files exist")
    
import sys

def turn_linear_to_logistic(steering, n_classes):
    interval = 2/n_classes
    classes = []
    lower_bound = -1
    
    for i in range(n_classes):
        upper_bound = lower_bound + interval if i < (n_classes-1) else 1
        classes.append(1 if steering > lower_bound and steering < upper_bound else 0)
        lower_bound += interval
        
    return np.array(classes)
    
    
break_classes = 21
X_train_left = None
y_train_left = None
X_train_right = None
y_train_right = None

X_train= []
y_train = []

with open('train.csv', 'r') as csv_file_train:
    
    csv_reader = csv.DictReader(csv_file_train)

    for row in csv_reader:
        X_train.append(row['path'])
        y_train.append(float(row['steering']))
        

X_validation = []
y_validation = []

with open('validation.csv', 'r') as csv_file_val:
    
    csv_reader = csv.DictReader(csv_file_val)

    for row in csv_reader:
        X_validation.append(row['path'])
        y_validation.append(float(row['steering']))
        
print('Loading done')


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Lambda, Dense, Activation, Flatten, Input, Dropout, Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

def get_model():
    input_tensor = Input(shape=(160, 320, 3))
    croped_input_img = Cropping2D(cropping=((60, 20), (0, 0)))(input_tensor)
    croped_input_img = Lambda(lambda x: (x / 255.0) - 0.5)(croped_input_img)
    base_model = VGG16(input_tensor=croped_input_img, weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    top_model = base_model.output
    top_model = Flatten()(top_model)
    top_model = Dropout(0.25)(top_model)
    
    top_model = Dense(2024)(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Activation('relu')(top_model)
    top_model = Dropout(0.25)(top_model)

    top_model = Dense(500)(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Activation('relu')(top_model)
    top_model = Dropout(0.25)(top_model)

    top_model = Dense(200)(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Activation('relu')(top_model)
    top_model = Dropout(0.25)(top_model)

    top_model = Dense(50)(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Activation('relu')(top_model)
    top_model = Dropout(0.25)(top_model)

    predictions = Dense(1)(top_model)

    model = Model(input=input_tensor, output=predictions)
    return model

def process_line(row):
    #b,g,r = cv2.split(cv2.imread(row['path']))
    #img = np.array(cv2.merge([r,g,b]))
    img = cv2.imread(row['path'])
    steering = float(row['steering'])
    return [img, steering]

def generate_arrays_from_file(path, batch_size = 20, flip=True):
    while 1:
        global X_train
        global y_train
        X_train, y_train = shuffle(X_train, y_train)
        Xs = []
        ys = []        
        for i in range(len(X_train)):
            if (len(Xs) == batch_size):
                yield (np.array(Xs), np.array(ys))
                Xs = []
                ys = []
                
            x, y = process_line({'path':X_train[i], 'steering':y_train[i]})
            Xs.append(x)
            ys.append(y)
            
            if flip:
                if (len(Xs) == batch_size):
                    yield (np.array(Xs), np.array(ys))
                    Xs = []
                    ys = []
                    
                x_flipped = np.fliplr(x)
                y_filpped = -y

                Xs.append(x_flipped)
                ys.append(y_filpped)

        yield (np.array(Xs), np.array(ys))

train_rows = len(X_train)
validation_rows = len(X_validation)
    
print('train records: ', train_rows)
print('validation records: ', validation_rows)

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint

model = get_model()
optimizer = RMSprop(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)
checkpoint = ModelCheckpoint('model2.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
history = model.fit_generator(
    generate_arrays_from_file('train.csv',batch_size=15), 
    samples_per_epoch=train_rows*2, 
    nb_epoch=30, 
    validation_data=generate_arrays_from_file('validation.csv', batch_size=10, flip=True),
    nb_val_samples=validation_rows*2, 
    callbacks=[checkpoint])