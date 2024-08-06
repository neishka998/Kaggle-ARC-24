import pandas as pd
from diffusers.models.autoencoders.vq_model import VQModel
from dataset import Arc24DatasetTransformations
import torch
import sys
from models.vq_vae import VQVAE
import torch.nn as nn

from torch.utils.data import DataLoader

def train_transformations(args): 

    vq_vae = VQVAE(
        input_channels=10, 
        output_channels=10, 
        hidden_dims=[16, 32, 64], 
        codebook_dim=128, 
        img_size=30, 
        latent_dim=64, 
        scale_factor=3, 
        expansion_factor=3
    )

    train_dataset = Arc24DatasetTransformations(
        data_path=r'C:\Users\tommy\Developer\Kaggle-ARC-24\arc-prize-2024\arc-agi_training_challenges.json' 
    )

    val_dataset = Arc24DatasetTransformations(
        data_path=r'C:\Users\tommy\Developer\Kaggle-ARC-24\arc-prize-2024\arc-agi_evaluation_challenges.json',
    )
    
    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=3, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)

    vq_vae.train_model(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        epochs=20,
        lr=3e-4,
        device='cuda'
    )

if __name__ == "__main__":
    train_transformations(None)