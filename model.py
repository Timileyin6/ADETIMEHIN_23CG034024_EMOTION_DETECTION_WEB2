# model.py
# Optional: example Keras training script template (requires tensorflow & dataset)
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# NOTE: This is a template. You must download FER2013 or prepare folders: train/val/test with subfolders per emotion.
DATA_DIR = "data/fer2013_small"  # your dataset path
emotion_model = load_model("_mini_XCEPTION.102-0.66.hdf5")

def build_model(input_shape=(48,48,1), num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'), layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # simple generator example (you must prepare images)
    gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = gen.flow_from_directory(DATA_DIR, target_size=(48,48), color_mode='grayscale', class_mode='categorical', subset='training', batch_size=64)
    val_gen = gen.flow_from_directory(DATA_DIR, target_size=(48,48), color_mode='grayscale', class_mode='categorical', subset='validation', batch_size=64)
    model = build_model(input_shape=(48,48,1), num_classes=train_gen.num_classes)
    model.fit(train_gen, validation_data=val_gen, epochs=20)
    model.save("creative_model_name_emotion.h5")
    print("Saved trained model to creative_model_name_emotion.h5")
