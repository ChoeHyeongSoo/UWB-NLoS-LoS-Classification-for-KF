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

epochs = 100
batch_size = 64
in_channels = 1
n_classes = 2
n_layers = 2
fc = 128

lr = 0.001
weight_decay = 0.0

# Parameters
view_train_iter = 100
view_val_iter = 5
save_point = 0.95

def get_clf_eval(y_true, y_pred, average='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, warn_for=tuple())
    return accuracy, precision, recall, f1

# Data Load ================================================================
df_uwb = data_process.df_uwb
df_uwb_data = data_process.df_uwb_data

x_train, x_test, y_train, y_test = train_test_split(df_uwb_data.values, df_uwb['NLOS'].values, test_size=0.1, random_state=42, stratify=df_uwb['NLOS'].values)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

print("x_train shape :", x_train.shape, y_train.shape)
print("x_val shape :", x_val.shape, y_val.shape)
print("x_test shape :", x_test.shape, y_test.shape)

print("Train NLOS 0 count :", len(y_train[y_train==0]))
print("Train NLOS 1 count :", len(y_train[y_train==1]))
print("Validation NLOS 0 count :", len(y_val[y_val==0]))
print("Validation NLOS 1 count :", len(y_val[y_val==1]))
print("Test NLOS 0 count :", len(y_test[y_test==0]))
print("Test NLOS 0 count :", len(y_test[y_test==1]))

def generating_loader(x_data, y_data, batch_size=batch_size, shuffle=True, drop_last=True):
    x_data = np.expand_dims(x_data, axis=1)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    # CIR CNN 처리하기 위한 전처리
    y_tensor = torch.tensor(y_data, dtype=torch.long).view(-1)

    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

trainloader = generating_loader(x_train, y_train, batch_size=batch_size, shuffle=True, drop_last=True)
validationloader = generating_loader(x_val, y_val, batch_size=batch_size, shuffle=False, drop_last=True)
testloader = generating_loader(x_val, y_val, batch_size=batch_size, shuffle=False, drop_last=True)

for x, label in trainloader:
    print(x.shape, label.shape)
    break

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size, num_layers, fully_connected, device):
        super(CNN_LSTM, self).__init__()
        self.batch_size = batch_size
        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=10, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        ) 
        self.lstm = nn.LSTM(input_size = 504, 
                            hidden_size = 32, 
                            num_layers = n_layers,
                            bias = False,
                            dropout = 0.5,
                            bidirectional = True,
                            batch_first=True)

        self.hidden_state, self.cell_state = self.init_hidden()
        
        self.bn = nn.BatchNorm1d(64)
        self.fc_layer = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc_layer_class = nn.Linear(128, out_channels)


    def init_hidden(self):
        hidden_state = torch.zeros(n_layers*2, self.batch_size, 32).to(device)
        cell_state = torch.zeros(n_layers*2, self.batch_size, 32).to(device)

        return hidden_state, cell_state
        
    def forward(self, x):
        x = self.conv1d_layer(x)
        x, _ = self.lstm(x,(self.hidden_state, self.cell_state))
        x = x[:, -1 :].view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc_layer(x)
        x = self.relu(x)
        x = self.fc_layer_class(x)
        x = self.relu(x)

        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(
    in_channels=in_channels,\
    device=device,\
    out_channels=n_classes,\
    batch_size=batch_size,\
    fully_connected=fc,\
    num_layers=n_layers).to(device)
loss_function = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # optimizer

start = time.time()

correct = 0
total = 0
train_acc = []
tmp_acc = 0
loss_arr = []

print("*Train Start!!*")
if torch.cuda.device_count() == True:
    print("epoch : {}, learing rate : {}, device : {}".format(epochs, lr, torch.cuda.get_device_name(0)))
else:
    print("epoch : {}, learing rate : {}, device : {}".format(epochs, lr, device))
print("Model : {}".format(model._get_name()))
print("Loss function : {}".format(loss_function._get_name()))
print("Optimizer : {}".format(str(optimizer).replace("\n", " ").replace("     ", ", ")))
print("*"*100)

correct = 0
total = 0
tmp_acc = 0
train_acc = []
loss_arr = []
val_view_acc = []
start = time.time()  # 시작 시간 저장

for epoch in range(epochs):
    epoch = epoch + 1
    for train_iter, (train_x, train_y_true) in enumerate(trainloader):
        model.train()  # Train mode
        model.zero_grad()  # model zero initialize
        optimizer.zero_grad()  # optimizer zero initialize
        
        train_x, train_y_true = train_x.to(device), train_y_true.to(device)  # device(gpu)
        train_y_pred = model.forward(train_x)  # forward

        loss = loss_function(train_y_pred, train_y_true)  # loss function
        loss.backward()  # backward
        optimizer.step()  # optimizer
        _, pred_index = torch.max(train_y_pred, 1)
    
        if train_iter % view_train_iter == 0:
            loss_arr.append(loss.item())
            total += train_y_true.size(0)  # y.size(0)
            correct += (pred_index == train_y_true).sum().float()  # correct
            tmp_acc = correct / total  # accuracy
            train_acc.append(tmp_acc.tolist())
            print("[Train] ({}, {}) loss = {:.5f}, Accuracy = {:.4f}, lr={:.6f}".format(epoch, train_iter, loss.item(), tmp_acc, optimizer.param_groups[0]['lr']))
    # validation 
    if epoch % view_val_iter == 0: 
        val_actual_tmp, val_pred_tmp = [], []
        for val_iter, (val_x, val_y_true) in enumerate(validationloader):
            model.eval()
            val_x, val_y_true = val_x.to(device), val_y_true.to(device)  # device(gpu)
            val_y_pred = model.forward(val_x)  # forward
            _, val_pred_index = torch.max(val_y_pred, 1)
            
            val_pred_index_cpu = val_pred_index.cpu().detach().numpy()
            val_y_true_cpu = val_y_true.cpu().detach().numpy()
            
            val_actual_tmp.append(val_pred_index_cpu.tolist())
            val_pred_tmp.append(val_y_true_cpu.tolist())
            
        val_acc, val_precision, val_recall, val_f1 = get_clf_eval(val_actual_tmp[1], val_pred_tmp[1])
        val_view_acc.append(val_acc)
        print("*[Valid] Accuracy:{:.4f}, Precison:{:.4f}, Recall:{:.4f}, F1 Score:{:.4f}".format(
             val_acc, val_precision, val_recall, val_f1))
print("Time : ", time.time()-start,'[s]',sep='')

model_save_path = 'Models/NLoS_Probability_Model.pth'
torch.save(model.state_dict(), model_save_path)


f = plt.figure(figsize=[8, 5])
f.set_facecolor("white")

plt.style.use(['default'])
plt.plot(loss_arr)
plt.plot(train_acc)


plt.legend(['Loss', 'Accuracy'])
plt.ylim((0.0, 1.05))
plt.grid(True)
plt.show()
