import tensorflow as tf
import time

from config import EPOCHS
from model import E2EImageCommunicator
from datasets import dataset_generator

# Reference: https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko

def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

train_ds = dataset_generator('./datasets/cifar10/train/')
test_ds = dataset_generator('./datasets/cifar10/test/')

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = E2EImageCommunicator(l=4)

@tf.function
def train_step(images):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(images, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  # train_accuracy(images, predictions)

@tf.function
def test_step(images):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(images, predictions)

  test_loss(t_loss)
  # test_accuracy(images, predictions)

# model.load_weights('./without_random_channel/epoch_200.ckpt')
# model.load_weights('./best_model/best_snr30')
# model.load_weights('./epoch_17.ckpt')

lowest_loss = 100

for epoch in range(1, EPOCHS+1):
  start_time = time.time()
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  # train_accuracy.reset_states()
  test_loss.reset_states()
  # test_accuracy.reset_states()

  for images, labels in train_ds:
    # images = tf.image.resize(images, [224, 224])
    train_step(images)
    
  for test_images, test_labels in test_ds:
    # test_images = tf.image.resize(test_images, [224, 224])
    test_step(test_images)

  print(
    f'Epoch {epoch}, '
    f'Loss: {train_loss.result():.6f}, '
    f'Test Loss: {test_loss.result():.6f}, '
    f'Training time: {(time.time() - start_time)/60:.2f}m'
  )

  # best model save
  if test_loss.result() < lowest_loss:
      lowest_loss = float(test_loss.result())
      model.save_weights(f'epoch_{epoch}.ckpt')
