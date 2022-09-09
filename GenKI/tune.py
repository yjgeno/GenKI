import torch
from .model import VGAE
from .train import VariationalGCNEncoder
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
from .preprocesing import split_data


hyperparams = {
"lr": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(1, 4))),
"beta": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(1, 5))),
"seed": tune.randint(0, 10000),
"weight_decay": tune.sample_from(lambda _: np.random.randint(1, 10)*(0.1**np.random.randint(3, 7)))
}


def train(config, checkpoint_dir = None): 
    train_data, val_data, test_data = split_data(dir = "data", load = True)
    torch.manual_seed(config["seed"]) 
    model = VGAE(VariationalGCNEncoder(train_data.num_features, 2))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])
    # optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"], momentum = 0.9, weight_decay = config["weight_decay"])
    
    # when restore a checkpoint
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(1000): # search < max_num_epochs
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        kl_loss = config["beta"] * model.kl_loss() # beta-VAE
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        # valid set
        with torch.no_grad():
            model.eval()
            z = model.encode(val_data.x, val_data.edge_index)
            auc, ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUROC: {auc:.4f}, AP: {ap:.4f}')

        # save a checkpoint
        with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)
        # record metrics from valid set
        tune.report(Loss = loss.item(), AUROC = auc, AP = ap, F_ = auc*ap) # call: shown as training_iteration
    print("Finished Training")


def test_best_model(best_trial):
    train_data, val_data, test_data = split_data(dir = "data", load = True)
    torch.manual_seed(best_trial.config["seed"])
    best_trained_model = VGAE(VariationalGCNEncoder(train_data.num_features, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_trained_model = best_trained_model.to(device)  
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device) 

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    # test set
    with torch.no_grad():
        best_trained_model.eval()
        z = best_trained_model.encode(test_data.x, test_data.edge_index)
        auc, ap = best_trained_model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    print(f"Best trial test AUROC: {auc:.4f}")
    print(f"Best trial test AP: {ap:.4f}")


def main(num_samples, 
        max_num_epochs = 100, 
        gpus_per_trial = 0,
        save_as: str = "tune_result"):
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
    print(f"Best trial config: {best_trial}")
    print("Best trial valid AUROC: {}".format(best_trial.last_result["AUROC"]))
    print("Best trial valid AP: {}".format(best_trial.last_result["AP"]))
    test_best_model(best_trial) # only should run once
    save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), f"{save_as}.csv")
    result.dataframe().to_csv(save_path)


if __name__ == "__main__":
    # seed for Ray Tune's random search algorithm
    np.random.seed(42)
    main(num_samples = 50)
    # report: https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics

