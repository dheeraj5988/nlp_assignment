import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("labeled_dataset.csv")
X = df['message']
y = df['satisfaction']

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

model = tf.keras.models.load_model("my_model.keras")
y_pred = model.predict(X_pad)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y, y_pred_labels))