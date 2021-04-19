# -*- coding: utf-8 -*-  
# =================================================

import numpy as np
import tensorflow as tf
from PIL import Image


"""

"""

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print("train_images: {}, train_labels: {}".format(train_images.shape, train_labels.shape))
print("test_images: {}, test_labels: {}".format(test_images.shape, test_labels.shape))

img = test_images[0, :].astype(np.uint8)

im = Image.fromarray(img)
im.save("./test_images/img_1.png")
