import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch import nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 2024 Season driver id
driver_lst = [830, 1, 846, 844, 857, 832, 815, 847, 4, 840, 807, 852, 817, 842, 825, 839, 848, 860, 855, 858]

# Desi implementation of feature scaling
final_driver_lst = []
for i in range(1,21):
  final_driver_lst.append(float(i))

driver_mapping = dict(zip(driver_lst, final_driver_lst))

with open('/Users/neel.agarkar/Documents/PROJECTS/F1 Prediction Model/driver_standings_new.csv', 'r') as file:
      # Reading data using csv library
    reader = csv.reader(file)
    data = list(reader)

      # Removing the string column which contains race positions and converting everything to float for tensor conversion
    column_to_remove = 5

    new_data_list = [row[:column_to_remove] + row[column_to_remove + 1:] for row in data]

    data_list_int = [[float(element) for element in row] for row in new_data_list]
    final_lst = [] 
    for i in range(len(data_list_int)):
      if data_list_int[i][2] in driver_lst:

        final_lst.append(data_list_int[i])

    data_array = np.array(final_lst)
    # replacing driver lst with an unskewed implementation
    for row in final_lst:
      if row[2] in driver_mapping:
        row[2] = driver_mapping[row[2]]


racer = int(input("Enter the Driver id: "))
race_start = int(input("Enter the number of races that you want to consider for the win prediction: "))


max_race_lst = []
for i in range(len(final_lst)):
  if final_lst[i][2] == racer:
    max_race_lst.append(final_lst[i])
max_race_tensor = torch.tensor(max_race_lst, dtype=torch.float32)
x, y = [], []
counter = 1
for i in range(race_start, len(max_race_lst)):
  x.append(counter)
  y.append(max_race_lst[i][4])
  counter += 1
plt.scatter(x, y)
plt.xlabel('Race Id')
plt.ylabel('Position')
x_array = np.array(x)
y_array = np.array(y)
plt.show()
train_split = int(0.8 * len(x)) # 80% of data used for training set, 20% for testing 
X_train, y_train = x_array[:train_split], y_array[:train_split]
X_test, y_test = x_array[train_split:], y_array[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))
print(X_train.shape, y_train.shape)
X_train = (X_train - np.mean(X_train)) / np.std(X_train)

model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(20, activation='softmax')  # Use softmax for probability distribution
], name='model0')



model.compile(
    loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for multi-class classification
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2  # Use a validation set to monitor training
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# Predict probabilities for test set
predictions = model.predict(X_test)
print(predictions[1])