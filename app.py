from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Flatten
import sys
print(sys.executable)

vgg16_obj = VGG16(include_top = False, input_shape = (224,224,3))    #  include_top = False is used to skip the layer from flattern
for layer in vgg16_obj.layers:             # Off the training of the trainable parameters
    layer.trainable = False

vgg16_obj.output
f1 = Flatten()(vgg16_obj.output)
final_layer = Dense(58, activation='softmax')(f1)
final_layer
model = Model(inputs=vgg16_obj.input,outputs=final_layer)


traffic_datagen = ImageDataGenerator(rescale=1/255,
                                  shear_range=0.7,
                                  zoom_range=0.5)
path=r"C:/Users/user/OneDrive/Desktop/traffic_sign_frontend/traffic_Data/DATA"
traffic_data =traffic_datagen.flow_from_directory(directory=path,target_size=(224,224),batch_size=3,class_mode="categorical",)
traffic_data.class_indices
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(traffic_data, epochs=10)
model.save('traffic_sign_vgg16.h5')

