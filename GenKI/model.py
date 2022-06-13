import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE
import torch.utils.tensorboard as tb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import get_distance


class VariationalGCNEncoder(torch.nn.Module):  # encoder
    def __init__(self, in_channels, out_channels, hidden=2):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden * out_channels)
        self.conv_mu = GCNConv(hidden * out_channels, out_channels)
        self.conv_logstd = GCNConv(hidden * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def train_VGAEmodel(
    data, out_channels=2, epochs=100, lr=0.01, log_dir="log_dir", verbose=True
):
    if log_dir is not None:
        train_logger = tb.SummaryWriter(os.path.join(log_dir, "train"))
    torch.manual_seed(42)
    # parameters
    num_features = data.num_features

    # model
    model = VGAE(VariationalGCNEncoder(num_features, out_channels))

    # to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 1e-6)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)

        recon_loss = model.recon_loss(z, data.edge_index)
        kl_loss = model.kl_loss()
        # print('recon_loss: {:.4f}'.format(float(loss)), 'kl_loss: {:.4f}'.format(float(kl_loss)))
        # loss = recon_loss + 0.3 * kl_loss
        loss = recon_loss + (1 / data.num_nodes) * kl_loss
        if log_dir is not None:
            train_logger.add_scalar("recon_loss", float(recon_loss), global_step)
            train_logger.add_scalar("kl_loss", float(kl_loss), global_step)
            train_logger.add_scalar("loss", float(loss), global_step)
        loss.backward()
        optimizer.step()
        global_step += 1
        if verbose:
            print("Epoch: {:03d}, loss: {:.4f}".format(epoch, float(loss)))
    return model


def save_model(model, name: str):
    """
    model: a trained VGAE model.
    name: str, name of .th file that will be saved in model.
    """
    if isinstance(model, VGAE):
        os.makedirs("./model", exist_ok=True)
        print(f'save model parameters to "./model/{name}.th"')
        return torch.save(model.state_dict(), f"./model/{name}.th")
    else:
        raise ValueError(f"model type {str(type(model))} not supported")


def load_model(obj, name: str, out_channels=2):
    """
    obj: a sc object.
    name: str, name of .th file saved in model.
    """
    r = VGAE(VariationalGCNEncoder(obj._counts.shape[0], out_channels))
    print(f'load model parameters from "./model/{name}.th"')
    r.load_state_dict(
        torch.load(f"./model/{name}.th", map_location=torch.device("cpu"))
    )
    return r


# after training
def get_latent_vars(data, model, plot_latent_mu=False):
    """
    data: torch_geometric.data.data.Data, WT data.
    model: a trained VGAE model.
    plot_latent_z: bool, whether to plot nodes with random sampled latent features.
    """
    torch.manual_seed(42)
    model.eval()
    z = model.encode(data.x, data.edge_index)  # feed and get latent
    z_m = model.__mu__.detach().numpy()
    z_S = (model.__logstd__.exp() ** 2).detach().numpy()  # variance
    if plot_latent_mu:
        # z_np = z.detach().numpy()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
        if z_m.shape[1] == 2:
            ax.scatter(z_m[:, 0], z_m[:, 1], s=4)
        elif z_m.shape[1] == 1:
            ax.hist(z_m, bins=60)
        elif z_m.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D

            ax = Axes3D(fig)
            ax.scatter(z_m[:, 0], z_m[:, 1], z_m[:, 2], s=4)
        plt.show()
    if z_m.shape[1] == 1:
        z_m = z_m.flatten()
        z_S = z_S.flatten()
    return z_m, z_S


def pmt(data, data_v, model, n=100, by="KL"):
    """
    data: torch_geometric.data.data.Data, WT data.
    data_v: torch_geometric.data.data.Data, virtual data.
    model: a trained VGAE model from WT data.
    n: int, # of permutation.
    by: str, method for distance calculation of distributions.
    """
    # permutate cells order and compute KL div
    n_cell = data.x.shape[1]
    dis_p = []
    np.random.seed(0)
    for _ in tqdm(range(n), desc="Permutating", total=n):
        # p: pmt, v: virtual, m: mean, S: sigma
        idx_pmt = np.random.choice(
            np.arange(n_cell), size=n_cell
        )  # bootstrap cell labels
        data_WT_p = Data(
            x=torch.tensor(data.x[:, idx_pmt], dtype=torch.float),
            edge_index=data.edge_index,
        )
        data_KO_p = Data(
            x=torch.tensor(data_v.x[:, idx_pmt], dtype=torch.float),
            edge_index=data_v.edge_index,
        )  # construct virtual data (KO) based on pmt WT
        z_mp, z_Sp = get_latent_vars(data_WT_p, model)
        z_mvp, z_Svp = get_latent_vars(data_KO_p, model)
        if by == "KL":
            dis_p.append(get_distance(z_mvp, z_Svp, z_mp, z_Sp))  # order KL
        # if by == 'reverse_KL':
        #     dis_p.append(get_distance(z_mp, z_Sp, z_mvp, z_Svp)) # reverse order KL
        if by == "t":
            dis_p.append(get_distance(z_mvp, z_Svp, z_mp, z_Sp, by="t"))
        if by == "EMD":
            dis_p.append(get_distance(z_mvp, z_Svp, z_mp, z_Sp, by="EMD"))
    return np.array(dis_p)
