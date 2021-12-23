from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import BATCH_SIZE

def dataset_generator(dir):
    dataset = image_dataset_from_directory(
        directory=dir,
        label_mode='int',
        labels='inferred',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(32, 32),
        shuffle=True,
        interpolation='bilinear'
    )

    return dataset
