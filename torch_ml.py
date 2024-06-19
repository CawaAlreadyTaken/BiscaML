import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ReteNeurale(nn.Module):
    def __init__(self, input_size):
        super(ReteNeurale, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def prepara_dati():
    df = pd.read_csv('database_partite.csv')
    X = df[['carta']].values
    y = df['indovinato'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), \
           torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

def train_model(X_train, y_train, input_size):
    model = ReteNeurale(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = outputs.squeeze().round()
        accuracy = (predicted == y_test).float().mean()
        print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepara_dati()
    input_size = X_train.shape[1]
    model = train_model(X_train, y_train, input_size)
    evaluate_model(model, X_test, y_test)

