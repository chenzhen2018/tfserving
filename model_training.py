# -*- coding: utf-8 -*-  
# =================================================

"""
2. Model Prepare:
    Train a neural network model to classify image of clothing, like sneakers and shirts.
    Finally, output a '.pb' model
"""

import tensorflow as tf

# =========================
# ===== Load dataset ======
# =========================
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print("train_images: {}, train_labels: {}".format(train_images.shape, train_labels.shape))
print("test_images: {}, test_labels: {}".format(test_images.shape, test_labels.shape))

# =========================
# ====== Preprocess =======
# =========================
train_images = train_images / 255.0
test_images = test_images / 255.0

# =========================
# ==== Build the model ====
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# =========================
# ==== Train the model ====
# =========================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# =========================
# ==== Save the model ====
# =========================
model.save('./saved_model/clothing/1/', save_format='tf')  # save_format: Defaults to 'tf' in TF 2.X, and 'h5' in TF 1.X.