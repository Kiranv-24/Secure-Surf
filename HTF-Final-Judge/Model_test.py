import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

os.makedirs("graphs", exist_ok=True)

# Data cleaning
def clean_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.fillna(0)

# Load dataset
phish = pd.read_csv('original_new_phish_25k.csv', dtype=str, low_memory=False)
legit = pd.read_csv('legit_data.csv', dtype=str, low_memory=False)
phish['Label'] = 1
legit['Label'] = 0
df = pd.concat([phish, legit])
df = df.drop(['url', 'NonStdPort', 'GoogleIndex', 'double_slash_redirecting', 'https_token'], axis=1)
df = clean_data(df)
X = df.drop('Label', axis=1)
y = df['Label'].astype(int)

# Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

input_dim = X_train.shape[1]

# Deep learning models
class BasicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class BatchNormMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

class TanhMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 128)
        self.fc_res1 = nn.Linear(128, 128)
        self.fc_res2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu(self.fc_in(x))
        res = self.fc_res2(self.fc_res1(x1))
        x2 = self.relu(x1 + res)
        return self.sigmoid(self.out(x2))

deep_classifiers = {
    'Basic MLP': BasicMLP,
    'Dropout MLP': DropoutMLP,
    'Deep MLP': DeepMLP,
    'BatchNorm MLP': BatchNormMLP,
    'Tanh MLP': TanhMLP,
    'Residual MLP': ResidualMLP,
}

# Train & Eval
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            true.append(yb.numpy())
    preds = np.vstack(preds)
    true = np.vstack(true)
    return accuracy_score(true, (preds > 0.5).astype(int))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []
model_objects = {}

best_model_instance = None
best_model_name = ""
best_acc = 0

# Train all deep models
for name, ModelClass in deep_classifiers.items():
    print(f"Training {name}...")
    model = ModelClass().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_weights = None
    best_model_acc = 0
    patience, patience_counter = 3, 0

    for epoch in range(20):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_model(model, test_loader, device)
        print(f"{name} - Epoch {epoch+1}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_model_acc:
            best_model_acc = val_acc
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    results.append((name, best_model_acc))
    model_objects[name] = model

    if best_model_acc > best_acc:
        best_acc = best_model_acc
        best_model_instance = model
        best_model_name = name

# DataFrame for accuracies
results_df = pd.DataFrame(results, columns=["Classifier", "Accuracy"])

# Bonferroni-Dunn test
try:
    dunn_results = sp.posthoc_dunn(results_df, val_col='Accuracy', group_col='Classifier', p_adjust='bonferroni')
except Exception as e:
    print("Dunn test error:", e)
    dunn_results = None

# Accuracy barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='Classifier', y='Accuracy', data=results_df, palette='viridis')
plt.xticks(rotation=45)
plt.title("Classifier Accuracies (Deep Learning)")
plt.tight_layout()
plt.savefig("graphs/classifier_accuracies_pytorch.png")
plt.show()

# Dunn heatmap
if dunn_results is not None:
    plt.figure(figsize=(8,6))
    sns.heatmap(dunn_results, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Bonferroni-Dunn Test (Deep Models)")
    plt.tight_layout()
    plt.savefig("graphs/dunn_heatmap.png")
    plt.show()

# Save best model
joblib.dump(best_model_instance.cpu(), "best_dl_model.joblib")
print(f"✅ Best model: {best_model_name} saved as 'best_dl_model.joblib' with accuracy {best_acc:.4f}")
