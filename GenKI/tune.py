import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from .model import VGAE
from .train import VariationalGCNEncoder
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pickle
import os


def save_gdata(data, name: str = "data"):
    with open(f'{name}.p', 'wb') as f:
        pickle.dump(data.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)


def _transform_data(data):
    transform = RandomLinkSplit(is_undirected = True, 
                                split_labels = True, 
                                num_val = 0.05, 
                                num_test = 0.1)
    return transform(data)


hyperparams = {
"lr": tune.loguniform(1e-4, 1e-1),
"beta": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(1, 5))),
"seed": tune.randint(0, 10000),
"weight_decay": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(2, 7)))
}

def train(config, checkpoint_dir = None):
    current_dir =  os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    data_path = os.path.join(current_dir, "data.p")
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    data = Data.from_dict(data)
    out_channels = 2

    torch.manual_seed(config["seed"])
    model = VGAE(VariationalGCNEncoder(data.num_features, out_channels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_data, val_data, test_data = _transform_data(data)
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])
    # optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"], momentum = 0.9, weight_decay = config["weight_decay"])
    
    for _ in range(1000): # search < max_num_epochs
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        kl_loss = config["beta"] * model.kl_loss() # beta-VAE
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            z = model.encode(val_data.x, val_data.edge_index)
            auc, ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUROC: {auc:.4f}, AP: {ap:.4f}')
        tune.report(Loss = loss.item(), AUROC = auc, AP = ap, F_ = auc*ap) # call: training_iteration
    print("Finished Training")


def main(num_samples, max_num_epochs = 200, gpus_per_trial = 0):
    scheduler = ASHAScheduler(
        max_t = max_num_epochs, # max iteration
        grace_period = 10, # stop at least after this iteration
        reduction_factor = 2)
    result = tune.run(train, 
                      config = hyperparams,
                      num_samples = num_samples, # trials: sets sampled from grid of hyperparams 
                      name = "experiment", # saved folder name
                      metric = "F_",
                      mode = "max",
                      resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
                      scheduler = scheduler, # prune bad runs 
                    #   stop = {'training_iteration':100}, # when tune.report was called
                      )
    best_trial = result.get_best_trial("F_", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final AUROC: {}".format(
        best_trial.last_result["AUROC"]))
    print("Best trial final AP: {}".format(
        best_trial.last_result["AP"]))
    save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "tune_result.csv")
    result.dataframe().to_csv(save_path)


if __name__ == "__main__":
    # seed for Ray Tune's random search algorithm
    np.random.seed(42)
    main(num_samples = 50)
    # how read report: https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics

