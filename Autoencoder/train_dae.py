#!/usr/bin/env python3 

# Owner: Akta Singh

import os, gc, random 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset 
from dense_autoencoder import DenseAutoencoder 


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED) 
random.seed(SEED) 
np.random.seed(SEED) 
torch.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 

# Selecting GPU
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters for dense Autoencoder
filename = "data"

input_dim = 45 
encoder_neurons = [32, 16, 8]
latent_dim = 6 
decoder_neurons = [8, 16, 32]

epochs = 300 
batch_size = 128 
learning_rate = 1e-3 

# Path to input data
path_to_file = "./data" 
X_train_file = "X_train.npy"
X_test_file  = "X_test.npy"
# Location to store data 
save_dir = f"{filename}_latent_{latent_dim}_batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}_optim_adam"

# Loading training & testing data from the files
X_train = np.load(os.path.join(path_to_file, X_train_file))
X_test  = np.load(os.path.join(path_to_file, X_test_file))

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(           
    TensorDataset(X_train, X_train), 
    batch_size=batch_size, 
    shuffle=True)

test_loader = DataLoader(          
    TensorDataset(X_test, X_test),
    batch_size=batch_size,
    shuffle=False)

# Model
model = DenseAutoencoder(
    input_dim,
    encoder_neurons,
    latent_dim,
    decoder_neurons
).to(device)

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Training & validation
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    model.train() 
    train_loss = 0.0 

    for x, _ in train_loader: 
        x = x.to(device) 

        optimizer.zero_grad() 
        recon, _ = model(x) 
        loss = criterion(recon, x) 
        loss.backward() 
        optimizer.step() 

        train_loss += loss.item() * x.size(0) 

    train_loss /= len(train_loader.dataset)  
    train_loss_history.append(train_loss) 
    print(f"Epoch [{epoch+1}/{epochs}]  Only Train Loss: {train_loss:.6f}", flush=True)

    model.eval() 
    val_loss = 0.0 
    with torch.no_grad(): 
        for x, _ in test_loader: 
            x = x.to(device) 
            recon, _ = model(x) 
            loss = criterion(recon, x) 
            val_loss += loss.item() * x.size(0) 

    val_loss /= len(test_loader.dataset) 
    val_loss_history.append(val_loss) 

    print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}", flush=True)

# Saving data
parameters = [input_dim, encoder_neurons, latent_dim, decoder_neurons] 
# needed to reconstruct the model when loading

model.save(
    save_dir, 
    parameters,
    np.array(train_loss_history),
    np.array(val_loss_history)
)

# Free the memory
del model, X_train, X_test 
gc.collect() 

