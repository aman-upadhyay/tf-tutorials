import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy', test_acc)
predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(0, 25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	prediction_label = np.argmax(predictions[i])
	true_label = test_labels[i]
	if prediction_label == true_label:
		color = 'green'
	else:
		color = 'red'
	plt.xlabel("{} ({})".format(class_names[prediction_label], class_names[true_label]), color=color)
plt.show()
