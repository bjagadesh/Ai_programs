import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example text data: list of sentences
sentences = [
    "This is a positive sentence.",
    "Negative sentiment detected.",
    "Neural networks are amazing.",
    "Another negative comment",
    "I love my family"
]

# Training data
labels = np.array([1, 0, 1, 0,1])  # 1 for positive, 0 for negative

# Create a vocabulary and encode sentences as sequences of word indices
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences to have equal length
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

# Create the neural network model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_sequence_length))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=70, batch_size=5)  # You can adjust epochs and batch_size

# Example new sentences for testing
new_sentences = [
    "This day is amazing",
    "Negative word detected",
    "I love this product.",
    "Some memories are bad"
]

# Preprocess the new sentences for prediction
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length, padding='post')

# Use the trained model for predictions
predictions = model.predict(new_padded_sequences)

# Interpret predictions
for i, sentence in enumerate(new_sentences):
    if predictions[i] >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    print(f"Sentence: '{sentence}' - Predicted Sentiment: {sentiment} (Probability: {predictions[i][0]:.4f})")
    #print("predictions of ",i," ",predictions[i])
