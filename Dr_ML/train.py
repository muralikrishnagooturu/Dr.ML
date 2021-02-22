from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense



img_width, img_height = 256, 256

folder_Peach_train = r'C:\Users\VENKATA KRISHNA\Desktop\murali\Dr_ML-20210218T072350Z-001\Dr_ML\DATA\train'

folder_Peach_valid = r'C:\Users\VENKATA KRISHNA\Desktop\murali\Dr_ML-20210218T072350Z-001\Dr_ML\DATA\validate'

nb_train_samples = 550
nb_validation_samples = 50
nb_epoch = 10

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        folder_Peach_train,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        folder_Peach_valid,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        #samples_per_epoch=3566,
        epochs=2,
        validation_data=validation_generator)
        #nb_val_samples=nb_validation_samples)




print('......')
#model.load_weights('first_try.h5')

#--------------------------TESTING--------------------------------------------

import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r"C:\Users\VENKATA KRISHNA\Desktop\murali\Dr_ML-20210218T072350Z-001\Dr_ML\DATA\train\benign\3.jpg",target_size=(256,256))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
validation_generator.class_indices
if result[0][0]>=0.5:
    print('Effected')
else:
    print('Not effected')





from keras.models import model_from_json
from keras.models import load_model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")
