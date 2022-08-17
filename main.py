import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 29
hidden_size = 5
num_classes = 4
num_epochs = 10
batch_size = 5
learning_rate = 0.001

#importing education data
df = pd.read_csv (r'student-mat.csv')
df.drop(['school'], axis=1, inplace=True)

#converting categorical data to numbers
df.replace({'sex':{'F':0, 'M':1}, 'address':{'U':0, 'R':1},
            'famsize':{'GT3': 0, 'LE3':1}, 'Pstatus':{'A':0, 'T':0},
            'Mjob': {'at_home':0, 'health':1, 'teacher':2, 'services':3,
                     'other': 4},
            'Fjob': {'at_home':0, 'health':1, 'teacher':2, 'services':3,
                     'other': 4},
            'reason':{'course': 0, 'home':1, 'reputation':2, 'other':3},
            'guardian':{'mother':0, 'father':1, 'other':2},
            'schoolsup':{'yes': 0, 'no':1}, 'famsup':{'yes':0, 'no':1},
            'paid':{'yes':0, 'no':1}, 'activities': {'yes':0, 'no':1},
            'nursery':{'yes':0, 'no':1}, 'higher':{'yes':0, 'no':1},
            'internet':{'yes':0, 'no':1}, 'romantic':{'yes':0, 'no':1}},
           inplace=True)

#Grades in the dataset range from 0-20; organizing the grades to have 4 classifications
for index, row in df.iterrows():
  if df.at[index, 'G3'] > 15:
    df.at[index, 'G3'] = 0
  elif df.at[index, 'G3'] > 10:
    df.at[index, 'G3'] = 1
  elif df.at[index, 'G3'] > 5:
    df.at[index, 'G3'] = 2
  else:
    df.at[index, 'G3'] = 3

# Checking if the data has been cleaned correctly
#print(df.head())

# Removing the grades so they are not used as inputs
g1, g2, g3 = df['G1'], df['G2'], df['G3']
df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)

train_data = df

# Setting the the training data size to 80% and testing size to 20%
X_train, X_val, y_train, y_val = train_test_split(df, g3, test_size=0.20, random_state=0)

# Custom Dataset
class FinalGradesDataset(Dataset):
    def __init__(self, X, Y):
        X = X.copy()
        self.x = X.copy().values.astype(np.float32)
        self.y = Y.copy().values.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# creating train and test datasets
train_ds = FinalGradesDataset(X_train, y_train)
test_ds = FinalGradesDataset(X_val, y_val)

# Data loader
# Shuffle is true for training
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

# Shuffle is false, not relevant for evaluation
test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

# Checking whether the batch size and input size are correct
examples = iter(train_loader)
inputs, label= examples.next()
print(inputs.shape, label.shape)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Creating the layers
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # No activation and no softmax at the end, because cross entropy loss already applies it
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # Uses softmax
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop for the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    # Loops over the epochs, all the batches
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad() # Empties the values in the gradient
        loss.backward() # Does backpropagation
        optimizer.step() # Updates the parameters

        #Prints loss, every nth step will print info about the process
        if (i + 1) % 32 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 35 test students: {acc} %')
