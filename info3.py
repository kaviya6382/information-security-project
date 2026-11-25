# Paste these cells between the two starter cells and the final test cell
# of the freeCodeCamp notebook. The starter cells already import numpy, pandas,
# tensorflow, keras, layers and load the data into `dataset`
# (columns: age, sex, bmi, children, smoker, region, expenses).


# ======================================================================
# Cell A: convert categorical data to numbers and split 80 / 20
# ======================================================================
# one-hot encode the text columns (dtype=float so Keras gets plain numbers)
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], dtype=float)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# pop the target off both sets
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')


# ======================================================================
# Cell B: build, compile and train the model
# ======================================================================
# scale every input column to mean 0 / std 1 (learned from the training data only)
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# the final test cell unpacks loss, mae, mse, so ask for both metrics
model.compile(loss='mae',
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['mae', 'mse'])

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=25, restore_best_weights=True)

history = model.fit(
    train_dataset,
    train_labels,
    epochs=300,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop])

print('Epochs trained:', len(history.history['loss']))
