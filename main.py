train_dir = 'D:/projects/brain-tumor-detection/dataset/Training'
test_dir = 'D:/projects/brain-tumor-detection/dataset/Testing'


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation & Rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(128,128,3)),   # <--- Add Input layer first
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)





model.save('D:/projects/brain-tumor-detection/models/brain_tumor_cnn.keras')


import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


loss, acc = model.evaluate(test_generator)
print(f"Final Test Accuracy: {acc*100:.2f}%")


from tensorflow.keras.models import load_model

loaded_model = load_model('D:/projects/brain-tumor-detection/models/brain_tumor_cnn.keras')

# Evaluate again to confirm
loss, acc = loaded_model.evaluate(test_generator)
print(f"Loaded Model Test Accuracy: {acc*100:.2f}%")



