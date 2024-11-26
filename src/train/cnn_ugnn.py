import sys
import os
import shutil
script_dir = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(script_dir))
project_name = "cnn_ugnn"
sys.path.append(project_path)


from torch.utils.data import DataLoader
from torch import optim 
from torch import nn
import pandas as pd
import torch
from tqdm import tqdm


from src.config.cnn_ugnn_config import CnnGegnConfig
from src.model.cnn_ugnn import CnnGegn
from src.utils.dataset import GegnDataset
import util
from src.preprocessor.base_preprocessor import (CoordPreprocessor, IdPreprocessor)
from src.preprocessor.cnn_ugnn_preprocessor import (
    preprocess,
    sample_every_epoch,
    test_preprocess,
    UgnnGridPreprocessor,
    UgnnEdgePreprocessor
)
from src.utils.model_util import get_score_in_scope


def setup_model(model):
    print("Model name:", config.model_name)
    print("Model version:", config.model_version)

    if torch.cuda.is_available():
        print("Device:", config.device)
        model = model.to(config.device)

    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    
    # load all the things in the saved model
    if config.reuse_model:
        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        print(step)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = checkpoint["best_loss"]
    else:
        for name, param in model.named_parameters():
            if ".bn" not in name:
                if ".weight" in name:
                    torch.nn.init.kaiming_normal_(
                        param.data, mode="fan_out", nonlinearity="leaky_relu"
                    )
                else:
                    torch.nn.init.constant_(param.data, 0)
    return model, criterion, optimizer


def train(p_data, model, edge_preprocessor):
    def save_model(path):
        checkpoint["epoch"] = epoch
        checkpoint["step"] = step
        checkpoint["model"] = model.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["best_loss"] = best_loss

        torch.save(checkpoint, path)
    
    def eval(best_loss):
        print('In eval')
        model.eval()
        
        data = (
            p_data["grid"]["grid_basic_dict"],
            p_data["grid"]["grid_extra_dict"],
            p_data["valid_edge"],
        )
        
        valid_dataset = GegnDataset(*data, config.one_hot_hour)
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.infer_batch_size, shuffle=False
        )
        valid_loader = tqdm(valid_loader)
        y_preds = []

        for i, valid_data in enumerate(valid_loader):
            torch.cuda.empty_cache()
            for j in range(len(valid_data)):
                valid_data[j] = valid_data[j].to(config.device)

            y_pred = model(*valid_data[:-1])
            y_preds.append(y_pred.cpu().detach())
            
            # train_dataloader.set_description(
            #         "Epoch: {} Step: {} Training Loss: {}".format(
            #             epoch, step, str(train_loss / (i_batch + 1))
            #         )
            #     )
            #     train_dataloader.refresh()
            

        y_pred = torch.cat(y_preds, dim=0)
        y_gt = torch.Tensor(p_data["valid_edge"][["trip"]].values).squeeze(-1)

        score = get_score_in_scope(y_pred, y_gt)
        print('total r_squared, mae, rmse, ssi, cpc, pearson: '+str(score)+'\n')

        eval_loss = criterion(y_pred, y_gt)
        if config.model_status == "train":
            # if config.save_tensorboard:
            #     writer.add_scalar("evaluation loss", eval_loss, global_step=step)
            #     writer.flush()
            save_model(config.MODEL_SAVE_PATH)
            if best_loss is None or best_loss > eval_loss:
                best_loss = eval_loss

                save_model(config.BEST_MODEL_SAVE_PATH)
                print("Best Model saved!")

            print()
        return best_loss
    
    model, criterion, optimizer= setup_model(model)

    parameters = {
        "model_version": config.model_version,
        "lr": config.learning_rate,
        "sample_type": config.sample_type,
        "beta": config.beta,
    }
    print(parameters)
    
    best_loss = None
    for epoch in range(config.epoch):
        torch.cuda.empty_cache()
        model.train()

        if epoch % config.sample_gap == 0:
            # We read the data and sample every epoch
            # Reading the data, then for every epoch sample it, takes too many mem.
            data = sample_every_epoch(
                edge_preprocessor, p_data, augmentation=config.do_augmentation
            )

            train_dataset = GegnDataset(*data, config.one_hot_hour)
            train_dataloader = DataLoader(
                train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=1
            )
        train_dataloader = tqdm(train_dataloader)
        train_loss = 0
        step = 0
        for i_batch, train_data in enumerate(train_dataloader):
            # o_grid_x, d_grid_x, edge_x, external_x, y = train_data
            # because there will be different data inside train_data, we will not assign the name for each of them
            if torch.cuda.is_available():
                for j in range(len(train_data)):
                    train_data[j] = train_data[j].to(config.device)

            y_pred = model(*train_data[:-1])

            loss = criterion(y_pred, train_data[-1])
            train_loss += loss.item()

            # if (i_batch + 1) % 10 == 0:
            train_dataloader.set_description(
                "Epoch: {} Step: {} Training Loss: {}".format(
                    epoch, step, str(train_loss / (i_batch + 1))
                )
            )
            # train_dataloader.refresh()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
        train_loss = train_loss / len(train_dataloader)

        # if config.save_tensorboard:
        #     writer.add_scalar("training loss", train_loss, global_step=self.step)
        #     writer.fllush()

        epoch += 1
        if (epoch + 1) % config.eval_gap == 0:
            if config.save_training_model:
                tqdm.write(f"epoch{epoch}, saving model in {config.MODEL_SAVE_PATH}.")
                save_model(config.MODEL_SAVE_PATH)
            best_loss = eval(best_loss)


def test(cities, model, note):
    if config.model_status == "test" and len(config.test_ids) != 0:
        test_df = test_df[test_df["o_id"].isin(config.test_ids)].reset_index(drop=True)
    elif config.model_status == "w_analysis":
        test_df = test_df[(test_df["o_id"].isin(config.w_analysis_ids) |
        (test_df["d_id"].isin(config.w_analysis_ids)))].reset_index(drop=True)

    md5 = util.calculate_md5(config.BEST_MODEL_SAVE_PATH)
    save_path = os.path.join(config.SAVE_PATH, 'cnn', md5)
    print(md5)
    # Create the folder path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Specify the destination path (including the file name)
    destination_path = os.path.join(save_path, "{}_{}_best.pt".format(config.model_name, config.model_version))

    # Copy the file to the destination path
    shutil.copy(config.BEST_MODEL_SAVE_PATH, destination_path)
    
    scores = []
    log = ""
    log += str(config.__dict__)
    log += '\n\n\n'
    
    
    for city in cities:
        log += 'City: '+city+'\n'
        preprocessor = make_preprocessor([city])
        p_data = test_preprocess(*preprocessor)

        parameters = {
            "model_version": config.model_version,
            "lr": config.learning_rate,
            "sample_type": config.sample_type,
            "beta": config.beta,
        }
        print(parameters)
        
        model.eval()
        
        test_dataset = GegnDataset(p_data['grid']['grid_basic_dict'], p_data['grid']['grid_extra_dict'], p_data['test_edge'], config.one_hot_hour)
        if config.model_status == "test":
            test_loader = DataLoader(test_dataset, batch_size=config.infer_batch_size,
                                        shuffle=False)
        elif config.model_status == "w_analysis":
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        test_loader = tqdm(test_loader)
        
        y_preds = []
        for i_batch, train_data in enumerate(test_loader):
            # o_grid_x, d_grid_x, edge_x, external_x, y = train_data
            # because there will be different data inside train_data, we will not assign the name for each of them
            if torch.cuda.is_available():
                
                for j in range(len(train_data)):
                    train_data[j] = train_data[j].to(config.device)
                
                y_pred = model(*train_data[:-1])
                y_preds.append(y_pred.cpu().detach())
        y_preds = torch.cat(y_preds, dim=0)
        y_preds[y_preds < 0] = 0.0
        result_df = pd.concat(
            [p_data['test_edge'][["city", "hour", "o_id", "d_id", "trip", "surface_distance"]],
                    pd.DataFrame(y_preds)], axis=1
        )
        result_df.columns = ["city", "hour", "o_id", "d_id", "gt_trip", "surface_distance", "pred_trip"]
        score = util.get_score(result_df, city)
        scores.append(score)
        
        score1 = get_score_in_scope(y_preds.squeeze(), torch.Tensor(result_df['gt_trip'].values).squeeze(-1))
        log += 'total r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score1)+'\n'
        score2 = get_score_in_scope(y_preds.squeeze(), torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [0, 10])
        log += '0, 10 r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score2)+'\n'
        score3 = get_score_in_scope(y_preds.squeeze(), torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [10, 100])
        log += '10, 100 r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score3)+'\n'
        score4 = get_score_in_scope(y_preds.squeeze(), torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [100, None])
        log += '> 100 r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score4)+'\n'
        log += '\n\n'

        result_df.to_csv(
            os.path.join(save_path, city+note+'.csv'),
            index=False)
        
        import pickle
        with open(os.path.join(save_path, city+note+'.pkl1'), 'wb') as file:
            # Use pickle to serialize the object and write it to the file
            pickle.dump(y_preds, file)
        print('pickle', os.path.join(save_path, city+note+'.pkl1'))

    with open(os.path.join(config.SAVE_PATH, 'cnn', md5, note), 'w') as file:
        file.write(log)
    
    
def make_preprocessor(cities):
    
    """
    Make preprocessor according to config.
    For training, the cities will involve various cities. However for testing,
    we load one dataset for one city, so the cities will ONLY contain one city.
    """
    
    coord_preprocessor = CoordPreprocessor(cities, config.CITY_CENTROID_PATH)
    id_preprocessor = IdPreprocessor(cities, config.split_frac)
    edge_preprocessor = UgnnEdgePreprocessor(
        cities, config.ALL_EDGE_PATH, checkpoint, config.one_hot_hour, config.hour
    )
    
    grid_preprocessor = UgnnGridPreprocessor(
        checkpoint,
        cities,
        config.grid_type,
        config.GRID_BASIC_PATH,
        config.GRID_POI_AOI_PATH,
        config.GRID_LAND_USE_PATH,
        config.RAW_CITY_METRIC_PATH,
        config.CITY_DIVISION_PATH,
        config.grid_radius,
        config.model_name,
        config.one_hot_hour,
    )
    preprocessor = (
        coord_preprocessor,
        id_preprocessor,
        edge_preprocessor,
        grid_preprocessor,
    )
    return preprocessor


def main():    
    if config.model_status == "train":
        print("🚋ing Model...")
        print(f"🚋 cities: {config.train_cities}")        
        
        cities = config.train_cities
        preprocessor = make_preprocessor(cities)
        p_data = preprocess(*preprocessor)
        model = CnnGegn(
            p_data["grid"]["grid_basic_dim"],
            p_data["grid"]["grid_extra_dim"],
            p_data["edge_dim"],
            p_data['grid']["external_dim"],
            config,
        )
        edge_preprocessor = preprocessor[2]  # We need to sample edge every epoch, thus load it into train func.
        train(p_data, model, edge_preprocessor)
        
    elif config.model_status == "test":
        print("Testing Model...")
        print(f"Test cities: {config.test_cities}")
        
        
        cities = config.test_cities        
        preprocessor = make_preprocessor([cities[0]])
        p_data = test_preprocess(*preprocessor)
        
        model = CnnGegn(
        p_data["grid"]["grid_basic_dim"],
        p_data["grid"]["grid_extra_dim"],
        p_data["edge_dim"],
        p_data['grid']["external_dim"],
        config,
        )

        model, criterion, optimizer= setup_model(model)
        
        test(cities, model, config.note)
    else:
        raise ValueError(f"Unknown model_status: {config.model_status}")



if __name__ == "__main__":
    util.setup_seed(95)
    util.get_gpu_memory_usage()


    config = CnnGegnConfig()
    
    print("Using "+config.device+"\n\n")
    checkpoint = util.load_checkpoint(config)
    def write_cf2model():
        print("Writng config into model...")
        # A temperary function to load config into model for version control.
        checkpoint["config"] = config.__dict__
        torch.save(checkpoint, config.BEST_MODEL_SAVE_PATH)
        print(config.__dict__)
        print("Write finished!")
    # write_cf2model()
    main()
