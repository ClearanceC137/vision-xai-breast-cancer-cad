
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# open log file
#log_file = open("Training_And_Validation_Advanced.txt", "w")

class Logger(object):
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)   # still print to console
        self.file.write(message)       # also save to file
        self.file.flush()              # make sure it writes immediately

    def flush(self):
        # needed for Python compatibility
        self.terminal.flush()
        self.file.flush()

# Redirect stdout (print statements)
#sys.stdout = Logger(sys.stdout, log_file)

# ------------------------------------------------------
# CNN Model Class with Debug Diagnostics + Max Pooling
# ------------------------------------------------------
# ------------------------------------------------------
# CNN Model
# ------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes,
                 conv_layers=[(32, 3), (64, 3)],
                 hidden_units=[256, 128],
                 dropout_rate=0.3,
                 leaky_alpha=0.01):
        super(CNNModel, self).__init__()

        H, W, C = input_shape  # expecting (H, W, C)
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = C
        for out_channels, ksize in conv_layers:
            self.convs.append(nn.Conv2d(in_channels, out_channels, ksize, padding=1))
            self.pools.append(nn.MaxPool2d(2))
            in_channels = out_channels

        # compute flatten size dynamically
        dummy = torch.zeros(1, C, H, W)
        for conv, pool in zip(self.convs, self.pools):
            dummy = pool(F.leaky_relu(conv(dummy), negative_slope=leaky_alpha))
        flatten_size = dummy.view(1, -1).size(1)

        # fully connected layers
        layers = []
        in_units = flatten_size
        for units in hidden_units:
            layers.append(nn.Linear(in_units, units))
            layers.append(nn.LeakyReLU(leaky_alpha))
            layers.append(nn.Dropout(dropout_rate))
            in_units = units

        # output layer
        layers.append(nn.Linear(in_units, num_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x expected shape: [B, H, W, C] → rearrange to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        for conv, pool in zip(self.convs, self.pools):
            x = pool(F.leaky_relu(conv(x)))
        x = x.reshape(x.size(0), -1)   # safer than view
        return self.fc(x)




# ------------------------------------------------------
# Training Loop
# ------------------------------------------------------
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device="cuda", save_path="trained_model/cnn_model_Advanced.pth"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0  # Track best validation accuracy
    history = []   # 🔹 keep metrics for all epochs
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            # Move batch to device
            X, y = X.to(device), y.to(device)


            # Forward + backward
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"[EPOCH {epoch+1}] Loss={total_loss/len(train_loader):.4f}, Acc={train_acc:.4f}")

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                out = model(X)
                _, preds = torch.max(out, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        print(f"[VAL] Acc={val_acc:.4f}")
        # -----------------------------
        # Save metrics
        # -----------------------------
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "val_acc": val_acc
        })

        # -----------------------------
        # Save model if validation improves
        # -----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f" Saved best model at epoch {epoch+1} with val_acc={val_acc:.4f}")
            
    return history ,best_val_acc

def load_trained_model(json_path, weight_path):
    """
    Loads a trained CNN model from a JSON config file and a .pth weight file.

    Args:
        json_path (str): Path to the JSON configuration file.
        weight_path (str): Path to the trained model weights (.pth file).
        CNNModelClass (class): The CNN model class definition used during training.

    Returns:
        torch.nn.Module: The loaded CNN model (set to eval mode).
    """
    # --- Load the JSON configuration ---
    with open(json_path, "r") as f:
        config = json.load(f)

    # --- Extract model and dataset parameters ---
    input_shape  = tuple(config["dataset"]["input_shape"])
    num_classes  = config["dataset"]["num_classes"]
    conv_layers  = [tuple(layer) for layer in config["model"]["conv_layers"]]
    hidden_units = config["model"]["hidden_units"]
    dropout_rate = config["model"]["dropout_rate"]

    # --- Select device ---
    device = "cuda" if torch.cuda.is_available() else config["training"]["device"]

    # --- Initialize model ---
    model = CNNModel(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_layers=conv_layers,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )

    # --- Load weights ---
    try:
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f" Model loaded successfully from '{weight_path}' on device: {device}")
    except FileNotFoundError:
        raise FileNotFoundError(f" Could not find weight file at '{weight_path}'")
    except Exception as e:
        raise RuntimeError(f" Failed to load model weights: {e}")

    return model
