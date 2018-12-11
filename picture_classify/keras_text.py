import cv2
import numpy as np
from keras.models import load_model
import data_sets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import time
train_path = 'D:\\project_file\\train_data\\smoke_car\\test\\'
image_size = 224
classes = ['no_smoke', 'smoke']
X,Y,_,cls = data_sets.load_train(train_path, image_size, classes)
X,Y,_,cls = shuffle(X,Y,_,cls)
Y = np.argmax(Y,1)
# predict = model.predict(X)
models = ['inceptionV3_1327model.h5','Vgg16model.h5','Vgg19model.h5']
model_name = ['InceptionV3','Vgg16','Vgg19']
scores = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
index = 0
# predict = data_sets.therhold_control(predict,score=0.4,index = 1)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
f,ax=plt.subplots(3,7,figsize=(32,15))
for i,model in enumerate(models):
    model = load_model(model)
    for j,score in enumerate(scores):
        start = time.time()
        predict = model.predict(X)
        predict = data_sets.therhold_control(predict, score=score, index=index)
        end = time.time()
        yongshi = end - start
        yongshi = round(yongshi,2)
        print(model_name[i]+' score : '+str(score)+' ：'+str(yongshi) + 's')
        # sns.heatmap(confusion_matrix(Y,predict),ax = ax[i,j],annot=True,fmt='2.0f',cmap='RdYlGn',linewidths=0.2,annot_kws={'size':50})
        # plt.tick_params(labelsize=23)
        sns.heatmap(confusion_matrix(Y, predict), ax=ax[i,j], annot=True,cmap='RdYlGn', fmt='2.0f')
        ax[i,j].set_title('模型名称 ：'+model_name[i]+'\n'+' 阈值 :' +str(score)+ ' index :'+ str(index) + ' 用时 :' + str(yongshi)+'s' ,fontsize = 10.0)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig('confusion_matrix0.jpg')
plt.show()