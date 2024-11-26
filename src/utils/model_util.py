import numpy
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np


from src.utils.metric import *

def get_edge_index(od_df, columns):
    id_df = pd.concat([od_df[columns[0]], od_df[columns[1]]], ignore_index=True)
    id_df = pd.DataFrame(id_df.unique(), columns=["id"]).reset_index()
    od_df = od_df.merge(id_df, left_on=columns[0], right_on="id")
    od_df = od_df.merge(id_df, left_on=columns[1], right_on="id")
    edge_index_df = od_df[["index_x", "index_y"]]
    return edge_index_df, id_df


def get_score_in_scope(y_pred, y_gt, scope=None, score=None):
    if scope is not None:
        if scope[0] is not None and scope[1] is not None:
            index = (y_gt >= scope[0]) & (y_gt < scope[1])
        elif scope[0] is not None:
            index = y_gt >= scope[0]
        elif scope[1] is not None:
            index = y_gt < scope[1]
        else:
            raise Exception
        y_pred = y_pred[index]
        y_gt = y_gt[index]
        if len(y_pred) == 0:
            print(f"scope{scope} has no value.")
            return None

    mae = nn.L1Loss()(y_pred, y_gt).item()
    rmse = RMSE()(y_pred, y_gt).item()
    ssi = SSI()(y_pred, y_gt).item()
    if ssi < 0:
        # torch.set_printoptions(profile="full")
        # print(y_pred, y_gt)
        numpy.savetxt("y_gt.csv", y_gt.numpy(), fmt='%e')
        numpy.savetxt("y_pred.csv", y_pred.numpy(), fmt='%e')
    cpc = CPC()(y_pred, y_gt).item()
    pearson = Pearson()(y_pred, y_gt).item()
    
    # Convert tensors to numpy arrays
    y_actual_np = y_gt.numpy()
    y_predicted_np = y_pred.numpy()

    # Calculate R-squared using scikit-learn
    r_squared = r2_score(y_actual_np, y_predicted_np)
    
    # Calculate SSIM

    # Calculate mean
    mean_actual = np.mean(y_actual_np)
    mean_predicted = np.mean(y_predicted_np)

    # Calculate standard deviation
    std_actual = np.std(y_actual_np)
    std_predicted = np.std(y_predicted_np)
    
    ssim = ((2*mean_actual*mean_predicted)*(2*std_actual*std_predicted))/((mean_actual**2 + mean_predicted**2)*(std_actual**2 + std_predicted**2))
    

    print("Scope:", scope)
    print("R-squared:", r_squared)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("SSI:", ssi)
    print("CPC:", cpc)
    print("Pearson:", pearson)
    print("SSIM", ssim)
    print()

    if score is not None:
        score.extend([r_squared, mae, rmse, ssi, cpc, pearson, ssim])
    return r_squared, mae, rmse, ssi, cpc, pearson, ssim