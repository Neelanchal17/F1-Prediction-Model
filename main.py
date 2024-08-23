import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch import nn

# 2024 Season driver id
driver_lst = [830, 1, 846, 844, 857, 832, 815, 847, 4, 840, 807, 852, 817, 842, 825, 839, 848, 860, 855, 858]

# Desi implementation of feature scaling
final_driver_lst = []
for i in range(1,21):
  final_driver_lst.append(float(i))

driver_mapping = dict(zip(driver_lst, final_driver_lst))

with open('driver_standings_new.csv', 'r') as file:
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

max_race_lst
max_race_lst = []
for i in range(len(final_lst)):
  if final_lst[i][2] == racer:
    max_race_lst.append(final_lst[i])
max_race_tensor = torch.tensor(max_race_lst, dtype=torch.float32)
x, y = [], []
for i in range(race_start, len(max_race_lst)):
  x.append(max_race_lst[i][1])
  y.append(max_race_lst[i][4])
plt.scatter(x, y)
plt.xlabel('Race Id')
plt.ylabel('Position')
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(x_tensor)
plt.show()
train_split = int(0.8 * len(x)) # 80% of data used for training set, 20% for testing 
X_train, y_train = x_tensor[:train_split], y_tensor[:train_split]
X_test, y_test = x_tensor[train_split:], y_tensor[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)
    # Forward defines the computation in the model
    def forward(self, X_test: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.linear_layer(X_test.view(-1, 1))


model_0 = LinearRegressionModel()

with torch.inference_mode(): 
    y_preds = model_0(X_test)
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")
print(model_0.parameters())


# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.1) 


# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
print(model_0.state_dict())