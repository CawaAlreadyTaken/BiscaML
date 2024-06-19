import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler

# Definisci il modello della rete neurale
class CardGameModel(nn.Module):
    def __init__(self):
        super(CardGameModel, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Funzione per caricare il modello
def load_model(model_path):
    model = CardGameModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Funzione per fare previsioni
def predict(model, opponent_cards, previous_declarations, scaler):
    # Prepara l'input
    input_data = opponent_cards + previous_declarations
    input_data = scaler.transform([input_data])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Fai la previsione
    with torch.no_grad():
        prediction = model(input_tensor)
        print(prediction)
        return prediction.item() >= 0.5

# Carica il modello
model_path = 'card_game_model.pth'
model = load_model(model_path)

# Esempio di input
opponent_cards = [10, 12, 8]
previous_declarations = [False, False]

# Standardizza i dati
scaler = StandardScaler()

# Dovresti adattare il modello di scaler ai tuoi dati reali, ad esempio utilizzando i dati di training
# Questo è solo un esempio e potrebbe richiedere un adattamento
df = pd.read_csv('dbs/format/format_db_4_2.csv')
df['z'] = df['z'].apply(ast.literal_eval)
df['k'] = df['k'].apply(ast.literal_eval)
X_opponent_cards = pd.DataFrame(df['z'].tolist(), index=df.index)
X_previous_declarations = pd.DataFrame(df['k'].tolist(), index=df.index)
X = pd.concat([X_opponent_cards, X_previous_declarations], axis=1).values
scaler.fit(X)

# Fai la previsione
result = predict(model, opponent_cards, previous_declarations, scaler)
print(f'La tua carta sarà più alta delle altre? {"Sì" if result else "No"}')

