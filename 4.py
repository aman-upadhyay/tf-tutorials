import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [
			tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
	except RuntimeError as e:
		print(e)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images1 = test_images
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

cnn_model = tf.keras.models.load_model('saved_model/modelcnn2')
predictions = cnn_model.predict(test_images)
y = predictions
plt.figure(figsize=(40, 40))
ni = 7
z = -1
for j in range(0, 1000):
	if (test_labels[j] == 1):
		if (z >= 4):
			break
		z = z + 1
		pass
	else:
		continue
	plt.subplot(ni, 11, 1 + z * 11)
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(test_images1[j], cmap='Greys')
	for i in range(0, 10):
		plt.subplot(ni, 11, z * 11 + i + 2)
		x = y[j, :, :, i]
		plt.imshow(x, cmap='Greys')
plt.show()
