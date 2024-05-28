import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import time
import random
import os
import uwb_dataset as uwb_dataset

columns, data = uwb_dataset.import_from_files()

for item in data:
	item[15:] = item[15:]/float(item[2])

print("\nColumns :", columns.shape, sep=" ")
print("Data :", data.shape, sep=" ")

cir_n = len(columns[15:])

print("Columns :", columns, sep=" ")
print("Channel Inpulse Response Count :", cir_n, sep=" ")

df_uwb = pd.DataFrame(data=data, columns=columns)
print("Channel 2 count :", df_uwb.query("CH == 2")['CH'].count())
print("Null/NaN Data Count : ", df_uwb.isna().sum().sum())
df_uwb.head(3)

los_count = df_uwb.query("NLOS == 0")["NLOS"].count()
nlos_count = df_uwb.query("NLOS == 1")["NLOS"].count()

print("Line of Sight Count :", los_count)
print("Non Line of Sight Count :", nlos_count)

df_uwb_data = df_uwb[["CIR"+str(i) for i in range(cir_n)]]
print("UWB DataFrame X for Trainndarray shape : ",df_uwb_data.values.shape)
print(df_uwb_data.head(5))

# # 입력데이터 시각화
# # 시간 기준 슬라이스 - Idx,Amp 형태 2차원 단면 이미지 형성
# image = df_uwb_data.iloc[0].values
# idx = range(len(image))
# plt.figure(figsize=(15, 6))
# plt.plot(idx, image)
# plt.title('CIR Visualization for First Sample')
# plt.xlabel('Idx')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()