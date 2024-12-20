import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AirQualityDataset(Dataset):
    def __init__(self, features, targets, sequence_length):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.sequence_length] #x is a sliding vector
        y = self.targets[idx + self.sequence_length] #y is a single number
        return torch.FloatTensor(x), torch.FloatTensor([y])


class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(BasicRNN, self).__init__()
#simplest version of RNN
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, #only apply dropout when number of alyers are more than one
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
        ) #we are interested in regression so output must be 1, and input must match RNN output which is the hidden layer

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # best default with hidden layer is 0, things can go wrong if you use random numbers 
        rnn_out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        last_time_step = rnn_out[:, -1, :]
        output = self.fc(last_time_step)
        return output


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()

            # Add gradient clipping to prevent exploding gradients (important for RNN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(
                    device
                ), batch_targets.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses


def main():
    # Hyperparameters
    sequence_length = 24 * 2        # past 2 days #can play around with hyperparameters
    hidden_size = 64                #he just picks random number doesnt mean they are the best.
    num_layers = 2                  #easiest change in model performance is to change learnign rate.
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    target = "C6H6(GT)"
    df = pd.read_excel("AirQualityUCI.xlsx") #might have to change a few things but this is basically the assingment code
    df = df.drop(["Date", "Time"], axis=1)

    df = df.replace(-200, np.nan)         # Replace -200 with NaN
    df = df.interpolate(method="linear")  # Interpolate missing values

    X, y = df.drop(target, axis=1), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #shuffle = false becausse of time dependancy (no data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = AirQualityDataset(X_train, y_train.values, sequence_length)
    test_dataset = AirQualityDataset(X_test, y_test.values, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X_train.shape[1]
    model = BasicRNN(input_size, hidden_size, num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device
    )

    print(f"Train Losses: {train_losses}")
    print(f"Validation Losses: {val_losses}")
    torch.save(model.state_dict(), "air_quality_rnn_model.pth")


if __name__ == "__main__":
    main()
