#data pre-processing

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2

# Define image size and categories
IMAGE_SIZE = 128
CATEGORIES = ['Black', 'Brown', 'Dark-brown', 'Olive', 'White']

def load_data(base_path):
    data = []
    labels = []
    for category in CATEGORIES:
        path = os.path.join(base_path, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                pass
    return np.array(data), np.array(labels)

# Load the dataset
base_path = 'skintone_images'
X, y = load_data(base_path)

# Normalize the data
X = X / 255.0

# One-hot encode the labels
y = to_categorical(y, num_classes=len(CATEGORIES))

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)




#Model train

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=50, verbose=1)
#save model
model.save('skin_tone_classifier.h5')




#Model Evaluation

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Print training and validation metrics
train_loss, train_accuracy = history.history['loss'][-1], history.history['accuracy'][-1]
val_loss, val_accuracy = history.history['val_loss'][-1], history.history['val_accuracy'][-1]

print(f"Training Loss: {train_loss}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

#results
#11/11 [==============================] - 5s 445ms/step - loss: 0.5533 - accuracy: 0.8023
#Test Loss: 0.5533307194709778
#Test Accuracy: 0.8022922873497009
#Training Loss: 0.4719012677669525
#Training Accuracy: 0.8099631071090698
#Validation Loss: 0.6416288614273071
#Validation Accuracy: 0.7586206793785095


#Visualization

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.show()





