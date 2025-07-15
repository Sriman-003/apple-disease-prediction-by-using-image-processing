import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

# Set dataset directories
TRAINING_DIR = "C:\\Users\\Admin\\OneDrive\\Desktop\\flask app\\apple_dataset\\Train"
TEST_DIR = "C:\\Users\\Admin\\OneDrive\\Desktop\\flask app\\apple_dataset\\Test"
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
MODEL_PATH = "best_model.h5"

if not os.path.exists(TRAINING_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Dataset directories not found. Please check your paths.")

# Manually defined class labels
class_labels = ["Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"]

# Data Generators
def train_val_generators(training_dir, testing_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        directory=training_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_labels  # Manually enforce class label order
    )
    
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    validation_generator = validation_datagen.flow_from_directory(
        directory=testing_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_labels  # Ensure test labels match
    )

    return train_generator, validation_generator

train_generator, validation_generator = train_val_generators(TRAINING_DIR, TEST_DIR)

# Verify Class Order Matches Manually Defined Labels
print("TensorFlow's detected class indices:", train_generator.class_indices)
print("Manually defined class labels:", class_labels)

# Ensure TensorFlow class mapping matches our manual labels
assert list(train_generator.class_indices.keys()) == class_labels, "Class order mismatch! Check folder names."

# Model with Transfer Learning
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')  # Ensure correct output size
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

# Train Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping]
)

# Plot Training & Validation Accuracy Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', linestyle='-')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot Training & Validation Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Evaluate Model
model = load_model(MODEL_PATH)  # Ensure best model is loaded
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)

# Use manually defined labels in evaluation
print(classification_report(validation_generator.classes, y_pred, target_names=class_labels))

# Generate and display confusion matrix
cm = confusion_matrix(validation_generator.classes, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Function for making predictions
def predict_image(image_path, model):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)
    predicted_index = np.argmax(prediction)

    # Use manually defined class labels
    predicted_class = class_labels[predicted_index]

    print(f"Prediction: {predicted_class}")
    return predicted_class

# Function to open file dialog and predict
def choose_file_and_predict():
    while True:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        root.destroy()
        
        if file_path:
            print(f"Selected file: {file_path}")
            predict_image(file_path, model)
        else:
            print("No file selected.")
        
        again = messagebox.askyesno("Select Another Image", "Do you want to select another image?")
        if not again:
            break

# Example Usage
if __name__ == "__main__":
    choose_file_and_predict()
