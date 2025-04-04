import pathlib
import pickle
import argparse
import os

import numpy as np
import torch
import pandas as pd

import veccs.orderings
from batram.legmods import Data, SimpleTM

def train_and_evaluate(train_size, save_dir="results"):
    """
    Train a model with a specified number of training samples and evaluate on test data.
    
    Parameters:
    -----------
    train_size : int
        Number of training samples to use
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    float
        Average log score across test samples
    """
    # Load locations and observations from CSV files
    locs = pd.read_csv("locs.csv", header=None).values  # Expect shape (4096, 2)
    obs = torch.as_tensor(pd.read_csv("stacked.csv", header=None).to_numpy(), dtype=torch.float32)
    obs = obs.T  # Now obs is (200, 4096): 200 images, each 4096 pixels

    ord = np.lexsort((locs[:, 1], locs[:, 0]))
    locs = locs[ord]
    obs = obs[:, ord]

    # Standardize observations
    obs_mean = obs.mean(dim=0, keepdim=True)
    obs_std = obs.std(dim=0, keepdim=True)
    obs = (obs - obs_mean) / obs_std

    # Split into train and test sets
    train_obs = obs[:160, :]
    test_obs = obs[160:200, :]

    # Use maximin ordering
    order = veccs.orderings.maxmin_cpp(locs)
    locs_ordered = locs[order, :]

    # Re-order observations according to maximin ordering
    train_obs_ordered = train_obs[:, order]
    test_obs_ordered = test_obs[:, order]

    # Subset training data to specified size
    curr_train_obs = train_obs[:train_size, :]
    curr_train_obs_ordered = curr_train_obs[:, order]

    # Find nearest neighbors
    largest_conditioning_set = 100
    nn = veccs.orderings.find_nns_l2(locs_ordered, largest_conditioning_set)

    # Create Data objects
    locs_tensor = torch.as_tensor(locs_ordered, dtype=torch.float32)
    curr_train_data = Data.new(locs_tensor, curr_train_obs_ordered, torch.as_tensor(nn))
    test_data = Data.new(locs_tensor, test_obs_ordered, torch.as_tensor(nn))

    # Create and fit the model
    print(f"Training model with {train_size} samples...")
    tm = SimpleTM(curr_train_data, theta_init=None, linear=False, smooth=1.5, nug_mult=4)
    
    # Set up optimizer
    nsteps = 200
    opt = torch.optim.Adam(tm.parameters(), lr=0.0001, weight_decay=4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps, eta_min=0.0001)
    
    # Fit the model
    res = tm.fit(nsteps, 0.0001, test_data=test_data, optimizer=opt, scheduler=sched, batch_size=64)
    
    n_test = test_obs.shape[0]
    log_scores = []
    print("N TEST: ", n_test)
    with torch.no_grad():  # Add this context manager
        for i in range(n_test):
            test_sample = test_obs_ordered[i]
            log_score = tm.score(test_sample)
            print(i, log_score.item())
            log_scores.append(log_score.item())    
    # Calculate average log score
    avg_log_score = np.mean(log_scores)
    print(f"Average log score with {train_size} training samples: {avg_log_score:.6f}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results
    result = {
        'train_size': train_size,
        'avg_log_score': avg_log_score,
        'log_scores': log_scores
    }
    
    with open(f"{save_dir}/result_n{train_size}.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    return avg_log_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a transport map model')
    parser.add_argument('--train_size', type=int, required=True, help='Number of training samples')
    parser.add_argument('--save_dir', type=str, default="results", help='Directory to save results')
    
    args = parser.parse_args()
    
    train_and_evaluate(args.train_size, args.save_dir)