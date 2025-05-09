import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from processing import create_generators
import matplotlib.pyplot as plt
import os

def create_advanced_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    # Load data
    train_gen, val_gen = create_generators()
    
    # Create model
    model = create_advanced_model()

    # Ensure output directory exists
    os.makedirs('models', exist_ok=True)

    # Callbacks - save as best_model.h5 to match predict.py
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[checkpoint, early_stop]
    )

    # Save the final model as well
    model.save('models/final_model.h5')
    print(f"Final model saved to models/final_model.h5")
    print(f"Best model saved to models/best_model.h5")

    # Save class indices for reference
    import json
    class_indices = train_gen.class_indices
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print(f"Class indices saved: {class_indices}")

    # Plotting training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Metrics')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Metrics')

    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()
    print("Training history plot saved to models/training_history.png")

if __name__ == '__main__':
    train_model()