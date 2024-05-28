import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

x = data_process.df_uwb_data.values
y = data_process.df_uwb['NLOS'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

in_channels = 1
out_channels = 2
batch_size = 64
num_epochs = 10
learning_rate = 0.01

def generating_loader(x_data, y_data, batch_size=batch_size, shuffle=True, drop_last=True):
    x_data = np.expand_dims(x_data, axis=1)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    # CIR CNN 처리하기 위한 전처리
    y_tensor = torch.tensor(y_data, dtype=torch.long).view(-1)

    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

train_loader = generating_loader(x_train, y_train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = generating_loader(x_val, y_val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = generating_loader(x_val, y_val, batch_size=batch_size, shuffle=False, drop_last=True)

# CNN Model ===============================================================
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=out_channels):
        super(CNN, self).__init__()
        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=10, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.bn = nn.BatchNorm1d(64)
        self.fc_layer = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc_layer_class = nn.Linear(128, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1d_layer(x)
        x = x[:, -1 :].view(x.size(0), -1)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = self.relu(x)
        x = self.fc_layer_class(x)
        x = self.softmax(x)
        return x

# 모델 인스턴스 생성
model = CNN(in_channels=in_channels, out_channels=out_channels)
loss_function = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_accuracy = correct_train / total_train

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            val_loss += loss.item()

            _, predicted_val = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    val_accuracy = correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

# 테스트
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

model_save_path = 'Models/CNN_Classifier.pth'
torch.save(model.state_dict(), model_save_path)
