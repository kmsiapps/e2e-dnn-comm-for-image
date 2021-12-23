import tensorflow as tf
import os

from config import EPOCHS, BATCH_SIZE
from model import E2EImageCommunicator, E2E_Channel, E2E_Decoder, E2E_Encoder
from datasets import dataset_generator

from qam_modem import qam_modem_awgn, qam_modem_rayleigh

# Reference: https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko

@tf.function
def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)
    image = tf.reshape(batch_images, (b, b, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (b*h, b*w, c))
    return image

def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

test_ds = dataset_generator('./datasets/cifar10/test/')

loss_object = tf.keras.losses.MeanSquaredError()
test_loss = tf.keras.metrics.Mean(name='test_loss')

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = E2EImageCommunicator(l=4)
model.build(input_shape=(1,32,32,3))
model.summary()

'''
for channelname in ['Rayleigh', 'AWGN']:
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    
    if not os.path.isdir(f'./results/{channelname}/'):
        os.mkdir(f'./results/{channelname}/')

    qam_modulation = qam_modem_rayleigh if channelname == 'Rayleigh' else qam_modem_awgn

    for EVAL_SNRDB in range(0, 45, 5):
        model = E2EImageCommunicator(l=4, snrdB=EVAL_SNRDB, channel='Rayleigh')
        # model.load_weights('./best_model/best_snr30')
        model.load_weights('./best_model/best_snr40')
        # model.load_weights('epoch_1.ckpt')

        i = 0
        ssim_props = 0; ssim_qams = 0
        mse_props = 0; mse_qams = 0
        for images, _ in test_ds:
            prop_results = model(images)

            # 256-QAM
            flat_images = tf.cast(tf.reshape(images, (-1)) * 255, 'uint8').numpy().tolist()
            demod_images = list(qam_modulation(x, 8, EVAL_SNRDB) for x in flat_images)
            qam_results = tf.reshape(tf.convert_to_tensor(demod_images, dtype=tf.float32) / 255, tf.shape(images))

            ssim_props += tf.reduce_sum(tf.image.ssim(images, prop_results, max_val=1.0))
            ssim_qams += tf.reduce_sum(tf.image.ssim(images, qam_results, max_val=1.0))

            mse_props += tf.reduce_sum(tf.math.sqrt((images - prop_results) ** 2))
            mse_qams += tf.reduce_sum(tf.math.sqrt((images - qam_results) ** 2))
            
            i += 1

            if i == 10:
                tf.keras.utils.save_img(f'./results/{channelname}/original_SNR{EVAL_SNRDB}.png', imBatchtoImage(images))
                tf.keras.utils.save_img(f'./results/{channelname}/proposed_SNR{EVAL_SNRDB}.png', imBatchtoImage(prop_results))
                tf.keras.utils.save_img(f'./results/{channelname}/256qam_SNR{EVAL_SNRDB}.png', imBatchtoImage(qam_results))
                break

        total_images = i * BATCH_SIZE
        ssim_props /= total_images
        ssim_qams /= total_images
        mse_props /= total_images
        mse_qams /= total_images
        
        print(f'Channel: {channelname} / SNR: {EVAL_SNRDB}dB =======================================')
        print(f'SSIM: (Proposed){ssim_props:.6f} vs. (QAM){ssim_qams:.6f}')
        print(f'MSE:  (Proposed){mse_props:.6f} vs. (QAM){mse_qams:.6f}')    

    # Layer-wise image
    images, _ = next(iter(test_ds))

    for model_subclass in [E2EImageCommunicator, E2E_Encoder, E2E_Channel, E2E_Decoder]:
        model = model_subclass(l=4, snrdB=10, channel='Rayleigh')
        model.load_weights('./best_model/best_snr40')
        result = model(images[:1, :, :, :])
        output = result
        if model_subclass.__name__ != 'E2EImageCommunicator':
            # For intermediate layers, flatten channels and normalize to grayscale images
            output = output / tf.reduce_max(output, axis=-1, keepdims=True)
            output = tf.transpose(output, (3, 0, 1, 2))
            output = tf.reshape(output, (-1, 32, 1))
        else:
            output = tf.reshape(output, (-1, 32, 3))
        tf.keras.utils.save_img(f'./results/{channelname}/{model_subclass.__name__}_SNR10.png', output)

        if model_subclass.__name__ == 'E2E_Encoder':
            tf.reshape(result, (-1)).numpy().tofile(f'./results/{channelname}/constellation.bin')

    tf.keras.utils.save_img(f'./results/{channelname}/E2E_before_SNR10.png',
                            images[0, :, :, :])
'''
