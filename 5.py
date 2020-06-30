# download data from https://drive.google.com/file/d/1lUp53MP8bauZCJ0LQJppn4wSKK-l6PZd/view?usp=sharingscap


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix
import os
from PIL import Image
from tqdm import tqdm_notebook as tqdm

############################################################################################

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [
			tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
	except RuntimeError as e:
		print(e)

############################################################################################

x = list()
y = list()

image_traini_dir = './city/train/image'
image_trainl_dir = './city/train/label'
image_testi_dir = './city/test/image'
image_testl_dir = './city/test/label'
image_traini_filename = os.listdir(image_traini_dir)
image_trainl_filename = os.listdir(image_trainl_dir)
image_testi_filename = os.listdir(image_testi_dir)
image_testl_filename = os.listdir(image_testl_dir)
for filename in range(1, 2796):
	image = Image \
		.open(os.path.join(image_traini_dir, str(filename) + '.jpg'))
	x.append(np.asarray(image))
for filename in range(1, 2796):
	image = Image \
		.open(os.path.join(image_trainl_dir, str(filename) + '.jpg'))
	y.append(np.asarray(image))
x = np.array(x) / 255
y = np.array(y)
y = y[:, :, :, np.newaxis]

train_features, train_labels = x, y
del x
del y
x = list()
y = list()
for filename in range(1, 501):
	image = Image \
		.open(os.path.join(image_testi_dir, str(filename) + '.jpg'))
	x.append(np.asarray(image))
for filename in range(1, 501):
	image = Image \
		.open(os.path.join(image_testl_dir, str(filename) + '.jpg'))
	y.append(np.asarray(image))
x = np.array(x) / 255
y = np.array(y)
y = y[:, :, :, np.newaxis]

test_features, test_labels = x, y
del x
del y

TRAIN_LENGTH = train_labels.shape[0]
BATCH_SIZE = 128
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset = train_dataset.shuffle(train_labels.shape[0]).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
test_dataset = test_dataset.shuffle(test_labels.shape[0]).batch(BATCH_SIZE)


def display(display_list):
	plt.figure(figsize=(15, 15))
	title = ['Input Image', 'True Mask', 'Predicted Mask']
	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i + 1)
		plt.title(title[i])
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		plt.axis('off')
	plt.show()


OUTPUT_CHANNELS = 2
base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

# Use the activations of these layers
layer_names = [
	'block_1_expand_relu',  # 64x64
	'block_3_expand_relu',  # 32x32
	'block_6_expand_relu',  # 16x16
	'block_13_expand_relu',  # 8x8
	'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
	pix2pix.upsample(512, 3),  # 8x8 -> 16x16
	pix2pix.upsample(256, 3),  # 16x16 -> 32x32
	pix2pix.upsample(128, 3),  # 32x32 -> 64x64
	pix2pix.upsample(64, 3),  # 64x64 -> 128x128
]


def unet_model(output_channels):
	inputs = tf.keras.layers.Input(shape=[256, 256, 3])
	x = inputs

	# Downsampling through the model
	skips = down_stack(x)
	x = skips[-1]
	skips = reversed(skips[:-1])

	# Upsampling and establishing the skip connections
	for up, skip in zip(up_stack, skips):
		x = up(x)
		concat = tf.keras.layers.Concatenate()
		x = concat([x, skip])

	# This is the last layer of the model
	last = tf.keras.layers.Conv2DTranspose(
		output_channels, 3, strides=2,
		padding='same')  # 128x128 -> 256*256

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


def create_mask(pred_mask):
	pred_mask = tf.argmax(pred_mask, axis=-1)
	pred_mask = pred_mask[..., tf.newaxis]
	return pred_mask[0]


def show_predictions(dataset=None, num=1):
	display([train_features[0], train_labels[0], create_mask(model.predict(train_features[0][tf.newaxis, ...]))])


def show_train_predictions(dataset=None, num=1):
	for image, mask in dataset.take(num):
		pred_mask = model.predict(image)
		display([image[0], mask[0], create_mask(pred_mask)])


show_predictions(dataset=False)


class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions()
		print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


EPOCHS = 20
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_train_predictions(test_dataset, 20)