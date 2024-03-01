import gym
import tqdm
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import optimizer

def train(learner, observations, actions, checkpoint_path, num_epochs=100):
    """Train function for learning a new policy using BC.
    
    Parameters:
        learner (Learner)
            A Learner object (policy)
        observations (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 11, ) 
        actions (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 3, )
        checkpoint_path (str)
            The path to save the best performing checkpoint
        num_epochs (int)
            Number of epochs to run the train function for
    
    Returns:
        learner (Learner)
            A Learner object (policy)
    """
    best_loss = float('inf')
    best_model_state = None

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.tensor(observations, dtype = torch.float32, device = device), torch.tensor(actions, dtype = torch.float32, device = device)) # Create your dataset
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True) # Create your dataloader

    # TODO: Complete the training loop here ###
    learner.train()
    for epoch in tqdm(range(num_epochs), desc="Training"):
        total_loss = 0
        for batch_index, (batch_observations, batch_actions) in enumerate(dataloader):
            optimizer.zero_grad()
            
            learner_actions = learner(batch_observations)
            loss = loss_fn(learner_actions, batch_actions)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)  # Calculate average loss per epoch
        # tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            
        # Saving model state if current loss is less than best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = learner.state_dict()
    
    # Save the best performing checkpoint
    torch.save(best_model_state, checkpoint_path)
    
    # Load the best model state
    learner.load_state_dict(best_model_state)
    
    return learner