import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def model():
    test_dir = r"C:\Users\zoepa\PycharmProjects\CarsClassifier\Cars Dataset\Test"
    train_dir = r"C:\Users\zoepa\PycharmProjects\CarsClassifier\Cars Dataset\Train"

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=25,
        validation_data=test_generator,
        validation_steps=test_generator.samples / test_generator.batch_size,
        verbose=1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model()

