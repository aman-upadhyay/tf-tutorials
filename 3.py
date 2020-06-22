import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


modelcnn = keras.Sequential()
modeldl = keras.Sequential()
modelcnn.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
modelcnn.add(keras.layers.MaxPooling2D((2, 2)))
modelcnn.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
modelcnn.add(keras.layers.MaxPooling2D((2, 2)))
modelcnn.add(keras.layers.Dropout(0.25))
modelcnn.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

modeldl.add(keras.layers.Flatten(input_shape=(3,3,64)))
modeldl.add(keras.layers.Dense(64, activation='relu'))
modeldl.add(keras.layers.Dense(10, activation='softmax'))
modelcnn.summary()
modeldl.summary()
model = tf.keras.Sequential([
  modelcnn,
  modeldl
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))



plt.plot(history.history['accuracy'], label='accuracy', color='orange',ls='-')
plt.plot(history.history['val_accuracy'], label = 'test accuracy', color='orange',ls='dotted')
plt.plot(history.history['loss'], label = 'loss', color='blue',ls='-')
plt.plot(history.history['val_loss'], label = 'test loss', color='blue',ls='dotted')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Test loss = {} and Test accuracy = {}.".format(test_loss, test_acc))

modelcnn.save('./saved_model/modelcnn')