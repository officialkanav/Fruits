import os
from os import listdir, makedirs
from os.path import join,exists,expanduser

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import Activation,Dropout,Flatten
from keras import backend as K
import tensorflow as tf

img_height, img_width = 244, 244
train_data_dir = "./fruits-360/Training"
validation_data_dir = "./fruits-360/Test"
batch_size = 16


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        class_mode='categorical')


inception_base = applications.ResNet50(weights='imagenet',include_top=False)

x = inception_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu')(x)

predictions = Dense(81,activation='softmax')(x)
model = Model(inputs=inception_base.input,outputs=predictions)


model.compile(optimizer = optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              epochs=5,
                              shuffle=True,
                              verbose=1,
                              validation_data=validation_generator)
model.save('my_model.h5')
model.save_weights('my_model_weights.h5')