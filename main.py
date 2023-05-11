import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf


def model():
    test_dir = r"C:\Users\zoepa\PycharmProjects\CarsClassifier\Cars Dataset\Test"
    train_dir = r"C:\Users\zoepa\PycharmProjects\CarsClassifier\Cars Dataset\Train"

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        # Batch size verkleinen als er error komt over memory
        batch_size=32,
        class_mode='categorical')

    # print(train_generator[0][0].shape)
    # print(train_generator[0][1].shape)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        # Batch size verkleinen als er error komt over memory
        batch_size=32,
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
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=3,
        validation_data=test_generator,
        validation_steps=test_generator.samples / test_generator.batch_size,
        verbose=1)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(val_loss)
    print(loss)

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.show()

    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples / test_generator.batch_size,
                                         verbose=1)
    print('Test accuracy:', test_acc)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model()

