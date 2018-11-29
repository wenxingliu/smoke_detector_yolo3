import tensorflow as tf
import numpy as np
import os, cv2
from PIL import ImageDraw,ImageFont
image_size = 256
num_channels = 3
images = []

path = "D:\\project_file\\train_data\\smoke_car\\test"
direct = os.listdir(path)
for file in direct:
    image = cv2.imread(path + '/' + file)
    print("adress:", path + '/' + file)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0 / 255.0)

font = ImageFont.truetype(font='font/FiraMono-Medium.otf')
list = []
for i,img in enumerate(images,start=1):
    x_batch = img.reshape(1, image_size, image_size, num_channels)

    sess = tf.Session()

    # step1网络结构图
    saver = tf.train.import_meta_graph('./smoke-model/smoke.ckpt-544.meta')

    # step2加载权重参数
    saver.restore(sess, './smoke-model/smoke.ckpt-544')

    # 获取默认的图
    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict_testing)

    res_label = ['no_smoke', 'smoke']
    x = res_label[result.argmax()]
    print('正在处理第'+str(i)+'/'+str(len(images))+'张图片')
    list.append(x)
print(list.count('smoke')/len(list))
