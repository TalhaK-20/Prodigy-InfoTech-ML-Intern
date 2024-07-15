import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Data loading and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Extract features using a pretrained model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Extracting features for training data
train_features = base_model.predict(train_generator)
train_labels = train_generator.classes

# Extracting features for validation data
validation_features = base_model.predict(validation_generator)
validation_labels = validation_generator.classes

# Flatten features
train_features_flat = train_features.reshape(train_features.shape[0], -1)
validation_features_flat = validation_features.reshape(validation_features.shape[0], -1)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(train_features_flat, train_labels)

# Predictions
y_pred = svm.predict(validation_features_flat)

# Evaluate the model
accuracy = accuracy_score(validation_labels, y_pred)
print(f'Accuracy: {accuracy}')
