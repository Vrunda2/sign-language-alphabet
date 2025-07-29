import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils.data_preparation import load_data

# Configuration
DATASET_DIR = "dataset"
EPOCHS = 30
BATCH_SIZE = 32
IMG_SIZE = (64, 64)

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("🔄 Loading data...")
    train_gen, val_gen = load_data(DATASET_DIR, IMG_SIZE, BATCH_SIZE)
    
    print("🏗️  Creating model...")
    model = create_model(train_gen.num_classes)
    model.summary()
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save class indices for later use
    with open('model/class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)
    print("✅ Class indices saved")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        "model/asl_model.h5", 
        save_best_only=True, 
        monitor='val_accuracy',
        verbose=1
    )
    early_stop = EarlyStopping(
        patience=10, 
        restore_best_weights=True, 
        monitor='val_accuracy',
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-7,
        verbose=1
    )
    
    print("🚀 Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    print("✅ Training completed!")
    print(f"✅ Best model saved at model/asl_model.h5")