import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

with open('./data.json', 'r') as f:
    data = json.load(f)

X = np.array([[v['area'], v['dis'], v['type'], v['middle_point'][0], v['middle_point'][1]] for v in data.values()])
y = np.array([v['price'] for v in data.values()])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x)

model = Net()
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

for epoch in range(2000):
    pred = model(X_train_tensor).squeeze()
    loss = nn.MSELoss()(pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        with torch.no_grad():
            train_pred = model(X_train_tensor).squeeze().numpy()
            test_pred = model(X_test_tensor).squeeze().numpy()
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            print(f"Epoch {epoch}: Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

with torch.no_grad():
    train_pred = model(X_train_tensor).squeeze().numpy()
    test_pred = model(X_test_tensor).squeeze().numpy()

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
print(f"Final - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

torch.save({'model': model.state_dict(), 'scaler': scaler}, 'model.pth')
print("Model saved!")
