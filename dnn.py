import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import os
import data_process as data_process

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
# Data Process : CIR amplitude distribution and distance. (#1 20107)
x = data_process.df_uwb_data.values
y = data_process.df_uwb['NLOS'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# Model&Parameter =======================================================
input_size = x_train.shape[1]
hidden_size_1 = 512
hidden_size_2 = 256
n_classes = 2
learning_rate = 0.01
num_epochs = 100

class DNN(nn.Module):
    def __init__(self, input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, num_classes=n_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

model = DNN(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, num_classes=n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train ===========================================================
model.train()
for epoch in range(num_epochs):
    inputs = torch.tensor(x_train, dtype=torch.float32).to(device)
    labels = torch.tensor(y_train, dtype=torch.int64).to(device)
    
    # Forward
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')   

# Test ======================================================================
model.eval()
with torch.no_grad():
    inputs = torch.tensor(x_test, dtype=torch.float32).to(device)
    labels = torch.tensor(y_test, dtype=torch.int64).to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    
    accuracy = accuracy_score(labels.cpu(), predicted.cpu())
    precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predicted.cpu(), average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

model_path = 'Models/dnn_classifier.pth'
torch.save(model.state_dict(), model_path)