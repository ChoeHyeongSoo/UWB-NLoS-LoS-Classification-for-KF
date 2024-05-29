from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Hyperparameter : 랜덤포레스트(분류분석)에 사용 기법 - GridSearchCV 
params = {
    'lr': [0.1, 0.01, 0.001],
    'max_epochs': [50, 100],
    'module__hidden_size': [256, 512, 1024],
    'optimizer': [optim.SGD, optim.Adam]
}

gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy')
gs.fit(x_train.astype(np.float32), y_train.astype(np.int64))

print("Best parameters: ", gs.best_params_)

# Test data eval
y_pred = gs.predict(x_test.astype(np.float32))
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')