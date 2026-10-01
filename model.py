import re
import pickle
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------
# Download required NLTK data
# ---------------------------------------------------------

nltk.download("stopwords")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("train.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# Remove missing values
# ---------------------------------------------------------

df = df.dropna()

print("Dataset after removing missing values:", df.shape)


# ---------------------------------------------------------
# Get input and output data
# ---------------------------------------------------------

X = df["title"]
y = df["label"]


# ---------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

corpus = []

print("Preprocessing text...")

for index, title in enumerate(X):

    if index % 1000 == 0:
        print(f"Processed {index} articles...")

    review = re.sub("[^a-zA-Z]", " ", str(title))

    review = review.lower()

    words = review.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    review = " ".join(words)

    corpus.append(review)


print("Text preprocessing completed.")


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

VOCAB_SIZE = 5000
MAX_LENGTH = 20

print("Creating tokenizer...")

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)

embedded_docs = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="pre"
)

print("Tokenization completed.")
print("Input shape:", embedded_docs.shape)


# ---------------------------------------------------------
# Convert data to NumPy arrays
# ---------------------------------------------------------

X_final = np.array(embedded_docs)
y_final = np.array(y)


# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.33,
    random_state=42,
    stratify=y_final
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# Create Bidirectional LSTM model
# ---------------------------------------------------------

EMBEDDING_DIM = 40

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
    )
)

model.add(
    Bidirectional(
        LSTM(100)
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ---------------------------------------------------------
# Compile model
# ---------------------------------------------------------

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64
)


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

print("\nEvaluating model...")

y_pred_probability = model.predict(
    X_test,
    verbose=0
)

y_pred = (
    y_pred_probability >= 0.5
).astype(int)


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(f"{accuracy:.4f}")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------

print("\nSaving model...")

model.save(
    "FakeNewsClassifier.keras"
)

print("Model saved as FakeNewsClassifier.keras")


# ---------------------------------------------------------
# Save tokenizer
# ---------------------------------------------------------

print("Saving tokenizer...")

with open(
    "tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )

print("Tokenizer saved as tokenizer.pkl")


print("\nTraining completed successfully!")
print("You can now run deploy.py")
