# Paste each block into the matching cell of the freeCodeCamp Colab notebook.
# Cells 1, 2, 4, 6, 9 and 11 are given to you; do not change them.
# Cell 2 defines: PATH, train_dir, validation_dir, test_dir, total_train,
# total_val, total_test, batch_size, epochs, IMG_HEIGHT, IMG_WIDTH.


# ======================================================================
# 3
# ======================================================================
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(
    directory=validation_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

# The test folder has no class subfolders, so point at its parent (PATH)
# and name the single "class" 'test'. shuffle=False keeps the order the
# final test cell expects; class_mode=None because the images are unlabeled.
test_data_gen = test_image_generator.flow_from_directory(
    directory=PATH,
    classes=['test'],
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode=None,
    shuffle=False)


# ======================================================================
# 5   (augmentation: 6 random transformations + the same rescale)
# ======================================================================
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# ======================================================================
# 7
# ======================================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')   # probability that the image is a dog
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# ======================================================================
# 8   (tip: set epochs = 30 in cell 2 for a better chance at 70%)
# ======================================================================
history = model.fit(
    x=train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size)


# ======================================================================
# 10
# ======================================================================
# Predict first (shuffle=False keeps the order), then grab the images to plot.
probabilities = [float(p) for p in model.predict(test_data_gen).flatten()]

test_data_gen.reset()
test_images = next(test_data_gen)

plotImages(test_images, probabilities=probabilities)
