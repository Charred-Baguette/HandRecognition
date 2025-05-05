import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras import layers
from keras.optimizers import Adam

image_size = (300, 300)
batch_size = 128

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training/train.weights.h5",
                                                 save_weights_only=True,
                                                 verbose=1)

print("Loading Dataset")
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "CNN/hands",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

"""
print("Loading model")
# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
])
"""

print("Loading model")
# Build a CNN model
def make_model(input_shape=(224, 224, 3)):

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


model = make_model((300, 300, 3))

print("compiling")
# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


print("training")
# Train the model
model.fit(train_ds, epochs=10, batch_size=64, validation_data=(val_ds), callbacks=[cp_callback])

# Evaluate the model
loss, accuracy = model.evaluate(val_ds)
print(f'Test accuracy: {accuracy}')