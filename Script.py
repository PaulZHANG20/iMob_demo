# coding:utf-8
##******************************************************************************//
##*            Intelligent Model Builder,iMoB (demo version)                   *//
##******************************************************************************//
##**  This script demos the application of neural network in device modeling  **//
## *****************************************************************************//
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Random Seed setting
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
# Path to data files (example data from ASAP7 tech files)
file_path = 'idvd.csv'
df1 = pd.read_csv(file_path, header=None)
idvd_data = df1.values
file_path = 'idvg.csv'
df2 = pd.read_csv(file_path, header=None)
idvg_data = df2.values
data = np.vstack([idvd_data, idvg_data])
# Remove data under zero bias
input_data = np.column_stack((data[:, 0], data[:, 1]))
output_data = data[:, 2]
input_data_filtered = input_data[input_data[:, 0] != 0]
output_data_filtered = output_data[input_data[:, 0] != 0]

# Model Objective: log10(Id/Vd) for zero-in-zero-out feature
output_data_transformed = np.log10(np.abs(output_data_filtered) / input_data_filtered[:, 0])
# Split the data into training and testing sets
input_train, input_test, output_train, output_test = train_test_split(
    input_data_filtered, output_data_transformed, test_size=0.1, random_state=42
)

input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
input_train_tensor = torch.tensor(input_train, dtype=torch.float32)
output_train_tensor = torch.tensor(output_train, dtype=torch.float32).unsqueeze(-1)
input_test_tensor = torch.tensor(input_test, dtype=torch.float32)
output_test_tensor = torch.tensor(output_test, dtype=torch.float32).unsqueeze(-1)


# MLP network configurations
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x


# Set up the model, loss function, and optimizer
input_size = 2
hidden_size1 = 5
hidden_size2 = 5
output_size = 1

model = MLP(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# training and testing datasets Setup
train_dataset = TensorDataset(input_train_tensor, output_train_tensor)
test_dataset = TensorDataset(input_test_tensor, output_test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Model Training Configration
num_epochs = 2500
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # The loss term is defined to improve the accuracy of the full bias, which can be defined according to different models and test data.
        loss = criterion(outputs, targets) + 6e9 * criterion(torch.pow(10, outputs) * inputs[:, 0].unsqueeze(-1),
                                                             torch.pow(10, targets) * inputs[:, 0].unsqueeze(-1))
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            test_loss = criterion(outputs, targets) + 6e9 * criterion(
                torch.pow(10, outputs) * inputs[:, 0].unsqueeze(-1),
                torch.pow(10, targets) * inputs[:, 0].unsqueeze(-1))

        test_losses.append(test_loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4e}, Test Loss: {test_loss.item():.4e}')

# Device Model Prediction
with torch.no_grad():
    model.eval()
    predictions = model(input_data_tensor)
vd_total = input_data_tensor[:, 0].unsqueeze(-1)
current_predicted = vd_total * np.power(10, predictions.numpy())
output_data = np.expand_dims(output_data, axis=1)
current_predicted = np.array(current_predicted)
relative_error = np.mean((current_predicted - output_data) / output_data)
with open('current_predicted.txt', 'w', encoding='utf-8') as f:
    np.savetxt(f, current_predicted, fmt='%e')
print('Training completed')
