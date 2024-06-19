import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ast import literal_eval

# Carica il dataset
df = pd.read_csv('dbs/db_4_2.csv')

# Converti le colonne di stringhe in liste
df['z'] = df['z'].apply(literal_eval)
df['k'] = df['k'].apply(literal_eval)

# Prepara i dati
X_opponent_cards = pd.DataFrame(df['z'].tolist(), index=df.index)
X_previous_declarations = pd.DataFrame(df['k'].tolist(), index=df.index)
y = df['e'].values

# Concatenare le colonne preparate
X = pd.concat([X_opponent_cards, X_previous_declarations], axis=1).values

# Dividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizza i dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converti i dati in tensori PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Definisci il modello della rete neurale
class CardGameModel(nn.Module):
    def __init__(self):
        super(CardGameModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Istanzia il modello, la funzione di loss e l'ottimizzatore
model = CardGameModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Allenamento del modello
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Valutazione del modello
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = y_pred.round()
    accuracy = (y_pred_class.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()
    print(f'Accuracy: {accuracy:.4f}')

torch.save(model.state_dict(), 'card_game_model.pth')
