from sklearn import datasets
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### Build dataset
class customed_dataset:
    def __init__(self, X, y, batch_size) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.maxNum = len(X)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index+self.batch_size < self.maxNum:
            self.index += self.batch_size
            return self.X[self.index-self.batch_size:self.index], self.y[self.index-self.batch_size:self.index]
        elif (self.index < self.maxNum) and (self.index+self.batch_size > self.maxNum):
            self.index += self.batch_size
            return self.X[self.index-self.batch_size:self.maxNum], self.y[self.index-self.batch_size:self.maxNum]
        else:
            raise StopIteration

def test(model, w, b,  loss_fun, dataset):
    loss = 0
    for dataset_X, dataset_y in dataset:
        y_head = model(w, b, dataset_X)
        dataset_y = np.array(dataset_y).flatten()
        loss += loss_fun(y_head, dataset_y)
    return loss

### Split the dataset
diabetes = datasets.load_diabetes()
X = pd.DataFrame(diabetes['data'], columns=diabetes['feature_names'])
X = X.iloc[:, :]
y = pd.DataFrame(diabetes['target'], columns=['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25, random_state=99)

### Build the simple linear model
def model(w, b, x):
    return np.dot(x, w)+b

# (y_head - y) ** 2
def loss_fun(y_head, y):
    return np.sum((y_head - y)**2)

### Feed the features into the model during training
epochs = 200
batch_size = 12
lr = 1e-2
iterations = math.ceil(len(X)/batch_size)
feature_size = X_train.shape[1]
np.random.seed(0)
w = np.random.rand(feature_size)
b = np.random.rand()

### Record
training_loss_record = np.array([])
validation_loss_reocrd = np.array([])
w_record = np.array([])
b_record = np.array([])

### Training model
for epoch in range(epochs):
    training_dataset = customed_dataset(X_train, y_train, batch_size)
    for dataset_X, dataset_y in training_dataset:
        y_head = model(w, b, dataset_X)
        dataset_y = np.array(dataset_y).flatten()
        loss = loss_fun(y_head, dataset_y)
        w -= np.dot(lr*2*(y_head-dataset_y), dataset_X) # 2 * (y_head - y) * x
        b -= np.sum(lr*2*(y_head-dataset_y)) # 2 * (y_head - y)
    
    ### Valiation
    training_dataset = customed_dataset(X_train, y_train, batch_size)
    validation_dataset = customed_dataset(X_val, y_val, batch_size)
    test_dataset = customed_dataset(X_test, y_test, batch_size)
    
    training_loss = test(model, w, b, loss_fun, training_dataset)
    training_loss_record = np.append(training_loss_record, training_loss)
    validation_loss = test(model, w, b, loss_fun, validation_dataset)
    validation_loss_reocrd = np.append(validation_loss_reocrd, validation_loss)
    
plt.title('The value of loss in the training process')
plt.xlabel('No. of epoch')
plt.ylabel('Loss')
plt.plot(training_loss_record, label='training loss')
plt.plot(validation_loss_reocrd, label='validation loss')
plt.legend()
plt.show()