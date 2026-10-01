import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. Define Image Dimensions
img_size = (224, 224)
batch_size = 32

# 2. Data Generators with Augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    'train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    'validation',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# 3. Load ResNet50 Base Architecture
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tuning setup: Freeze all layers except the last 15 layers
base_model.trainable = True
for layer in base_model.layers[:-15]:
    layer.trainable = False

# 4. Custom Top Classification Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)  # Dropout layer helps shield against overfitting
output = Dense(3, activation='softmax')(x)  # 3 Outputs: Fully Stocked, Low, Out of Stock

model = Model(inputs=base_model.input, outputs=output)

# 5. Compile and Train
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Shelf Stock Monitor Model...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# 6. Save Model Architecture natively
model.save('shelf_monitor_model.keras')
print("Model trained and saved as shelf_monitor_model.keras successfully!")