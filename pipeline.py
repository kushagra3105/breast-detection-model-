import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv(
    'https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv'
)

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64).view(-1, 1)

# Define a simple neural network model

class MySimpleNN(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        out = self.linear(features)
        out = self.sigmoid(out)
        return out
   
learning_rate = 0.1
epochs = 25
loss_function = nn.BCELoss()

model = MySimpleNN(X_train_tensor.shape[1])

oprtimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    loss = loss_function(y_pred, y_train_tensor)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    oprtimizer.zero_grad()
    loss.backward()
    oprtimizer.step()

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_test_tensor).float().mean()

print(f"Accuracy: {accuracy.item():.4f}")
