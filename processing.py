from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_generators():
    """
    Create and return data generators for training and validation.
    """
    # Check if dataset exists
    if not os.path.exists('dataset'):
        print("ERROR: Dataset directory not found!")
        print("Please run organize.py first to create the dataset structure.")
        return None, None
        
    # Check if classes exist
    if not (os.path.exists('dataset/compliant') and os.path.exists('dataset/non-compliant')):
        print("ERROR: Dataset classes not found!")
        print("Please run organize.py first to create the dataset structure.")
        return None, None
    
    # Create data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    # Create validation generator
    validation_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    print(f"Classes found: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")

    return train_generator, validation_generator

if __name__ == '__main__':
    train_gen, val_gen = create_generators()
    if train_gen and val_gen:
        print("Preprocessing setup complete!")
        print(f"Classes: {train_gen.class_indices}")
    else:
        print("Failed to create data generators. Please check error messages above.")