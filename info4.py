 Paste these cells between the two starter cells and the final test cell
# of the freeCodeCamp notebook. The starter cells already import tensorflow,
# pandas, numpy and download the files:
#   train_file_path / test_file_path  (tab-separated, no header: label, message)
 
 
# ======================================================================
# Cell A: load the data, turn labels into numbers (ham = 0, spam = 1)
# ======================================================================
from tensorflow import keras
from tensorflow.keras import layers
 
cols = ['label', 'message']
train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=cols)
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=cols)
 
train_labels = (train_df['label'] == 'spam').astype('float32').values
test_labels = (test_df['label'] == 'spam').astype('float32').values
 
BATCH = 32
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df['message'].values, train_labels)).shuffle(5000).batch(BATCH)
test_ds = tf.data.Dataset.from_tensor_slices(
    (test_df['message'].values, test_labels)).batch(BATCH)
 
 
# ======================================================================
# Cell B: text vectorization + model
# ======================================================================
vectorizer = layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100,
    standardize='lower_and_strip_punctuation')
vectorizer.adapt(train_df['message'].values)
 
model = keras.Sequential([
    keras.Input(shape=(), dtype=tf.string),
    vectorizer,
    layers.Embedding(vectorizer.vocabulary_size(), 32),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')     # probability of spam
])
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
 
 
# ======================================================================
# Cell C: train
# ======================================================================
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)
 
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    callbacks=[early_stop])
 
 
# ======================================================================
# Cell D: the prediction function
# ======================================================================
def predict_message(pred_text):
    prob = float(model.predict(tf.constant([pred_text]), verbose=0)[0][0])
    label = 'spam' if prob > 0.5 else 'ham'
    return [prob, label]
 
 
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)
 
