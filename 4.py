import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [
			tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
	except RuntimeError as e:
		print(e)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images1 = test_images
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
cnn_model = tf.keras.models.load_model('saved_model/modelcnn2')
predictions = cnn_model.predict(test_images)
y = predictions
for j in range(0,25):
  plt.figure(figsize=(20, 20))
  plt.subplot(1, 11, 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid('off')
  plt.imshow(test_images1[j], cmap=plt.cm.binary)
  prediction_label = np.argmax(predictions[j])
  true_label = test_labels[j]
  plt.xlabel("{}".format(class_names[true_label]))
  for i in range(0, 10):
    plt.subplot(1,11,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    x = y[j,:,:,i]
    plt.imshow(x, cmap=plt.cm.binary)
  plt.savefig('./plots/test_img{}'.format(j))
