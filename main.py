import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv("Proooject.csv")
data['Winning Hands'] = data['Winning Hands'].map({
  'Player': 0,
  'Banker': 1,
  'Tie': 2
})

# Create a classifier object with Monte Carlo Dropout
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

def monte_carlo_dropout_predict(X, model, n_samples):
    results = np.zeros((n_samples,))
    for i in range(n_samples):
        results += model.predict(X)
    results /= n_samples
    return results.argmax()

def update():
    global data
    # Update the model with the new data
    X_update = np.array(X_pred)
    Y_update = np.array([correct_outcome_mapped])
    model.partial_fit(X_update, Y_update, classes=[0, 1, 2])
    # Update the data DataFrame
    new_row = pd.DataFrame({'Winning Hands': [correct_outcome_mapped]})
    data = pd.concat([data, new_row], ignore_index=True)

def allocation():
  global X_train, Y_train, X, data
  X = data['Winning Hands']
  # Exclude NaN values from X_train and Y_train
  X_train = X.iloc[:-1].dropna()  # Take all values other than last
  Y_train = X.iloc[1:].dropna()  # shift the indices 1 ahead
  Y_train = Y_train.reset_index(
    drop=True)  # Reset Y_train indices (to remove error due to above 2 steps)

# Step 1: Load Preprocessed the data
allocation()

# Create a classifier object
model = PassiveAggressiveClassifier()

while True:
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X_train.values.reshape(-1, 1), Y_train, epochs=10)

  # Make a prediction using Monte Carlo Dropout
  n_samples = 10
  X_pred = np.array(X.iloc[-1]).reshape(1, -1)
  Y_pred = monte_carlo_dropout_predict(X_pred, model, n_samples)
  out = Y_pred

  # Get the predicted outcome
  out_array = ["Player", "Banker", "Tie"]
  print("Predicted outcome:", out_array[out])

  # Get the user feedback
  feedback = input("Was the prediction correct? (yes/no): ").lower()

  # If the feedback is no, update the model
  if feedback == "no":
    correct_outcome = input(
      "Enter the correct outcome (Player, Banker, or Tie): ").capitalize()
    correct_outcome_mapped = {
      'Player': 0,
      'Banker': 1,
      'Tie': 2
    }[correct_outcome]

    update()

  continue_prompt = input("Continue prediction? (yes/no): ").lower()

  if feedback == "yes":
    correct_outcome_mapped = out

    update()

    data.to_csv("Proooject.csv", index=False)

  allocation()

  if continue_prompt == "no":
    data.to_csv("Proooject.csv", index=False)
    break

# Map the numerical values back to their original labels
data['Winning Hands'] = data['Winning Hands'].map({
  0: 'Player',
  1: 'Banker',
  2: 'Tie'
})

# Save the updated data back to the CSV file
data.to_csv("Proooject.csv", index=False)
