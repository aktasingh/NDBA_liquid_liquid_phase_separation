#!/usr/bin/env python3 

# Owner: Akta Singh

import os 
import pickle
import torch 
import torch.nn as nn
import numpy as np


class DenseAutoencoder(nn.Module): 
"""
Dense Autoencoder using fully-connected layers.
"""
    def __init__(self,             
                 input_dim,        
                 encoder_neurons,  
                 latent_dim,       
                 decoder_neurons): 
        super().__init__() 

        # Encoder
        encoder_layers = [] 
        prev_dim = input_dim 
        for h in encoder_neurons: 
            encoder_layers.append(nn.Linear(prev_dim, h)) 
            encoder_layers.append(nn.BatchNorm1d(h)) 
            encoder_layers.append(nn.ReLU()) 
            prev_dim = h 

        encoder_layers.append(nn.Linear(prev_dim, latent_dim)) 

        self.encoder_net = nn.Sequential(*encoder_layers) 

        # Decoder
        decoder_layers = [] 
        prev_dim = latent_dim 
        for h in decoder_neurons: 
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.BatchNorm1d(h)) 
            decoder_layers.append(nn.ReLU()) 
            prev_dim = h 

        decoder_layers.append(nn.Linear(prev_dim, input_dim)) 

        self.decoder_net = nn.Sequential(*decoder_layers) 

    def encode(self, x): 
        return self.encoder_net(x) 

    def decode(self, z): 
        return self.decoder_net(z) 

    def forward(self, x): 
        z = self.encode(x) 
        x_hat = self.decode(z) 
        return x_hat, z 


    # Saving & loading the parameters, weights & saving loss of the autoencoder model 
    def save(self, save_dir, parameters, train_loss, val_loss): 
        os.makedirs(save_dir, exist_ok=True) 
       
        with open(os.path.join(save_dir, "parameters.pkl"), "wb") as f: 
            pickle.dump(parameters, f) 

        torch.save(self.state_dict(), os.path.join(save_dir, "weights.pth")) 

        np.savetxt(os.path.join(save_dir, "training_loss.dat"), train_loss) 
        np.savetxt(os.path.join(save_dir, "validation_loss.dat"), val_loss) 

    @classmethod 
    def load(cls, save_dir):
        # Load the parameters
        with open(os.path.join(save_dir, "parameters.pkl"), "rb") as f: 
            params = pickle.load(f) 

        model = cls(*params) 
        # Load the weights
        model.load_state_dict(torch.load(os.path.join(save_dir, "weights.pth"))) 
        model.eval() 
        return model 

