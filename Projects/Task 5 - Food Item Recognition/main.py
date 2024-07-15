import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'train',  # It should be the same directory as train with validation split
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the trained model
model.save('food_item_recognition_model.keras')

# Load the trained model
model = tf.keras.models.load_model('food_item_recognition_model.keras')

# Load test data CSV
test_data = pd.read_csv('Food_Data.csv')

# Predicting and displaying correct class name
try:
    # Select an image from the train folder
    img_path = 'train/class2/1e70914864.jpg'  # Update 'suborder_name' and image name

    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = list(train_generator.class_indices.keys())[predicted_class]

    # Get the correct class name from test_data.csv based on the ImageId
    correct_class_name = test_data.loc[test_data['ImageId'] == img_path.split('/')[-1], 'ClassName'].values
    if len(correct_class_name) > 0:
        correct_class_name = correct_class_name[0]
        print(f'Predicted Food: {correct_class_name}')
    else:
        print(f'Predicted label {predicted_label} not found in the CSV file.')

except Exception as e:
    print(f'Error: {e}')
