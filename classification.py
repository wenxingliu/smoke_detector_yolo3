from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense
import numpy as np
from data_preprocess import acquire_data_and_label_from_image


nosmoke_path = './data/train_data/No_smoke/'
smoke_path = './data/train_data/Smoke/'
nosmoke_train_data, nosmoke_label = acquire_data_and_label_from_image(nosmoke_path)
smoke_train_data, smoke_label = acquire_data_and_label_from_image(smoke_path)
data = np.concatenate((nosmoke_train_data,smoke_train_data), axis=0)
label = np.concatenate((nosmoke_label,smoke_label), axis=0)

vgg19 = VGG19(weights='imagenet')
base_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc1').output)
x = base_model.output

x = Dense(1024, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:19]:
    layer.trainable = False
for layer in model.layers[19:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.000001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
# train the model on the new data for a few epochs
model.fit(x=data, y=label, epochs=5, validation_split=0.2, shuffle=True)


