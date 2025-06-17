import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class Neural_Network_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(Neural_Network_Regressor, self).__init__()
        self.fully_connected_layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_size, output_size)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.is_trained = False
    def forward(self, x):
        x = self.fully_connected_layer_1(x)
        x = self.relu(x)
        x = self.fully_connected_layer_2(x)
        return x
        # return self.fully_connected_layer_2(self.relu(self.fully_connected_layer_1(x)))

    def train(self, X_tensor, y_tensor):
        num_epochs = 1000
        batch_size = 100
        class Regression_Dataset(Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
                if len(self.data) != len(self.labels): raise AssertionError("Data and labels must have same length!") 
            def __len__(self):                
                return len(self.data)
            def __getitem__(self, index):
                return self.data[index], self.labels[index]
        if len(y_tensor.shape) != len(X_tensor.shape):
            y_tensor = y_tensor.unsqueeze(1)
        dataset = Regression_Dataset(X_tensor, y_tensor)
        train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for index, (X_batch, y_batch) in enumerate(train_data_loader):
                
                predicted = self.forward(X_batch)
                loss = self.loss_function(predicted, y_batch)
                # Backpropagation
                self.optimizer.zero_grad() # should zero out in at before backpropagation at    each epoch
                loss.backward()
                self.optimizer.step()
                
            if (epoch + 1) % 100 == 0: print(f'Epoch [{epoch + 1} / {num_epochs}]; Loss : {loss.item():.4f}')
        self.is_trained = True
    def predict(self, X_test):
        if not self.is_trained: raise AssertionError("Must first train NN model!")

        return self.forward(X_test)

def toy_test_run():
    np.random.seed(0)
    X = np.random.rand(100, 1)
    y = 3 * X + 2 + np.random.rand(100, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    model = Neural_Network_Regressor(1, 10, 1)
    model.train(X_tensor, y_tensor)
    predicted = model.predict(X_tensor).detach().numpy()
    plt.scatter(X, y, label='Original Data')
    plt.plot(X, predicted, label='Fitted Line', color='red')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Regression using Neural Network')
    plt.show()


