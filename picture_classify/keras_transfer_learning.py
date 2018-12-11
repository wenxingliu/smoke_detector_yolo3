from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
import data_sets
from sklearn.model_selection import train_test_split

train_path = 'D:\\project_file\\train_data\\smoke_car\\trian\\'
image_size = 224
classes = ['no_smoke', 'smoke']
X,Y,_,_ = data_sets.load_train(train_path, image_size, classes)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
def base_model(x):
    if x == 'vgg16':
        base_model = VGG16(weights='imagenet')
    elif x == 'inceptionV3'
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif x == 'vgg19'
        base_model = VGG16(weights='imagenet')
    return base_model

base_model = base_model('vgg19')
x = base_model.output
# x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=25)
score ,acc = model.evaluate(X_test, y_test, batch_size=25)


model.save('Vgg16model.h5')