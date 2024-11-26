import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_path)



import pandas as pd
from src.utils.model_util import (
    get_score_in_scope,
)
import torch






def get_csv_filenames(directory):
    """
    Get a list of filenames in the given directory that end with '.csv'.

    Parameters:
    - directory (str): The path to the directory.

    Returns:
    - list: A list of filenames with the '.csv' extension.
    """
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    return csv_files

# Example usage:
grid1000path = '/home/lai/ugnn/results/gegn_outputs/cnn/be8cc673d0a396f5fc1b6054f6c43d9c'
grid1000s = get_csv_filenames(grid1000path)
log = ''
for grid1000 in grid1000s:
    print(grid1000)
    log += grid1000
    log += '\n'
    grid1000 = pd.read_csv(grid1000)
    groups = grid1000.groupby("o_id")
    for group in groups:
        gt_sum = sum(group[1]['gt_trip'])
        pred_sum = sum(group[1]['pred_trip'])
        if pred_sum == 0:
            pred_sum += 0.1
        new_val = group[1]['pred_trip']/pred_sum*gt_sum
        grid1000.loc[new_val.index, 'pred_trip'] = new_val

    score1 = get_score_in_scope(torch.Tensor(grid1000['pred_trip'].values).squeeze(-1), torch.Tensor(grid1000['gt_trip'].values).squeeze(-1))
    log += 'total r_squared, mae, rmse, cpc, pearson: '+str(score1)+'\n'
    score2 = get_score_in_scope(torch.Tensor(grid1000['pred_trip'].values).squeeze(-1), torch.Tensor(grid1000['gt_trip'].values).squeeze(-1), [0, 10])
    log += '0, 10 r_squared, mae, rmse, cpc, pearson: '+str(score2)+'\n'
    score3 = get_score_in_scope(torch.Tensor(grid1000['pred_trip'].values).squeeze(-1), torch.Tensor(grid1000['gt_trip'].values).squeeze(-1), [10, 100])
    log += '10, 100 r_squared, mae, rmse, cpc, pearson: '+str(score3)+'\n'
    score4 = get_score_in_scope(torch.Tensor(grid1000['pred_trip'].values).squeeze(-1), torch.Tensor(grid1000['gt_trip'].values).squeeze(-1), [100, None])
    log += '> 100 r_squared, mae, rmse, cpc, pearson: '+str(score4)+'\n'
    log += '\n\n'
with open(os.path.join(grid1000path, 'log_after_calibration1'), 'w') as file:
    file.write(log)

