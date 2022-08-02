import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from .model import VGAE
import torch.utils.tensorboard as tb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import get_distance


class VariationalGCNEncoder(torch.nn.Module):  # encoder
    def __init__(self, in_channels, out_channels, hidden = 2):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden * out_channels)
        self.conv_mu = GCNConv(hidden * out_channels, out_channels)
        self.conv_logstd = GCNConv(hidden * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VGAE_trainer():
    def __init__(self, 
                 data, 
                 out_channels: int = 2, 
                 epochs: int = 200, 
                 lr: float = 2e-3, 
                 log_dir: str = None, 
                 verbose: bool = True,
                 beta: str = 3e-3,
                 **kwargs):
        self.num_features = data.num_features
        self.out_channels = out_channels
        self.data = data 
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.logging = True if log_dir is not None else False
        if self.logging:
            self.train_logger = tb.SummaryWriter(os.path.join(log_dir, "train"))
            self.test_logger = tb.SummaryWriter(os.path.join(log_dir, "test"))      
        if beta is not None:
            self.beta = beta
        else:
            self.beta = (1 / self.train_data.num_nodes)    


    def __repr__(self) -> str:
        return f"Hyperparameters\n"\
               f"epochs: {self.epochs}, lr: {self.lr}, beta: {self.beta:.4f}\n"

    def _transformer(self, num_val = 0.05, num_test = 0.2, **kwargs):
        if self.verbose:
            print(f"Data split into Train ({1-num_val-num_test}), Valid ({num_val}), Test ({num_test})")
        return RandomLinkSplit(is_undirected = True, 
                                split_labels = True, 
                                num_val = num_val, 
                                num_test = num_test)

    # split data
    def _transform_data(self, x_noise: float = None, edge_noise: float = None, **kwargs):
        """
        Args:
        x_noise: Standard deviation of white noise added to the training data. Defaults to None.
        edge_noise: Remove or add edges to the training data.
        """
        from copy import deepcopy
        data_ = deepcopy(self.data)

        if x_noise is not None:
            if self.verbose:
                print(f"add white noise to data x, level: {x_noise} SD")
            data_.x = data_.x + x_noise * torch.randn(data_.x.shape)
            
        if edge_noise is not None:
            train_data_, _, _ = self._transformer(num_val = 0, num_test = 0)(data_) # upper diag of adjacent
            total = train_data_.pos_edge_label_index.shape[1]
            n = int(abs(edge_noise) * total)
            if edge_noise > 0:
                from torch_geometric.utils import negative_sampling        
                fake_edge_index = negative_sampling(train_data_.pos_edge_label_index,  # care for data leakage
                                                    num_neg_samples= n,
                                                    num_nodes = len(data_.x))
                fake_edge_index = torch.cat((fake_edge_index, fake_edge_index[torch.LongTensor([1, 0])]), 1) # swap, diagonal stays
                fake_edge_index = torch.unique(torch.cat((data_.edge_index, fake_edge_index), 1), dim = 1)        
                if self.verbose:
                    print(f"add noise to data edge, level: {edge_noise}: sampled {n} neg edges as pos")
                    print("(orig)", data_.edge_index.shape[1], f"+ 2 * (n) {n} = (edges to use)", fake_edge_index.shape[1])            
            else:
                weights = torch.tensor([1/total] * total, dtype = torch.float) # uniform weights
                if n > total:
                    raise ValueError("cannot retain edges more than the total")
                index = weights.multinomial(num_samples = n, replacement = False)
                fake_edge_index = train_data_.pos_edge_label_index[:, index]
                fake_edge_index = torch.cat((fake_edge_index, fake_edge_index[torch.LongTensor([1, 0])]), 1)
                if self.verbose:
                    print(f"sample data edge, ratio: {abs(edge_noise)}")
                    print("(orig)", data_.edge_index.shape[1], "(edges to use)", fake_edge_index.shape[1])        
            data_.edge_index = fake_edge_index
        self.train_data, self.val_data, self.test_data = self._transformer()(data_)


    # refer to https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
    def train(self, **kwargs):
        global_step = 0
        # torch.manual_seed(8096)
        self._transform_data(**kwargs)
        self.model = VGAE(VariationalGCNEncoder(self.num_features, self.out_channels))
        
        # to cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.train_data, self.val_data, self.test_data = self.train_data.to(device), self.val_data.to(device), self.test_data.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 9e-4)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum = 0.9, weight_decay = 5e-4)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            z = self.model.encode(self.train_data.x, self.train_data.edge_index)
            recon_loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index, self.train_data.neg_edge_label_index)
            kl_loss = self.beta * self.model.kl_loss() # beta-VAE
            loss = recon_loss + kl_loss
            if self.logging:
                self.train_logger.add_scalar("loss", loss.item(), global_step)
                self.train_logger.add_scalar("recon_loss", recon_loss.item(), global_step)
                self.train_logger.add_scalar("kl_loss", kl_loss.item(), global_step)
            loss.backward()
            optimizer.step()
            global_step += 1

            with torch.no_grad():
                self.model.eval()
                z = self.model.encode(self.test_data.x, self.test_data.edge_index)
                auc, ap = self.model.test(z, self.test_data.pos_edge_label_index, self.test_data.neg_edge_label_index)
                if self.logging:
                    self.test_logger.add_scalar("AUROC", auc, global_step)
                    self.test_logger.add_scalar("AP", ap, global_step)
                if self.verbose:
                    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, AUROC: {auc:.4f}, AP: {ap:.4f}")
        self.final_metrics = global_step, loss.item(), auc, ap


    def save_model(self, name: str):
        """
        name: str, name of .th file that will be saved in model.
        """
        if isinstance(self.model, VGAE):
            os.makedirs("model", exist_ok=True)
            path = os.path.join("model", f"{name}.th")
            print(f"save model parameters to {path}")
            return torch.save(self.model.state_dict(), f"model/{name}.th")
        else:
            raise ValueError(f"model type {type(self.model)} not supported")


    def load_model(self, name: str):
        """
        name: str, name of .th file saved in model.
        """
        r = VGAE(VariationalGCNEncoder(self.num_features, self.out_channels))
        path = os.path.join("model", f"{name}.th")
        print(f"load model parameters from {path}")
        r.load_state_dict(
            torch.load(os.path.join("model", f"{name}.th"), map_location=torch.device("cpu"))
        )
        self.model = r


    # after training
    def get_latent_vars(self, data, plot_latent_mu = False):
        """
        data: torch_geometric.data.data.Data.
        plot_latent_z: bool, whether to plot nodes with random sampled latent features.
        """
        self.model.eval()
        _ = self.model.encode(data.x, data.edge_index) 
        z_m = self.model.__mu__.detach().numpy()
        z_S = (self.model.__logstd__.exp() ** 2).detach().numpy()  # variance
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


    def pmt(self, data_v, n = 100, by = "KL"):
        """
        data_v: torch_geometric.data.data.Data, virtual data.
        n: int, # of permutation.
        by: str, method for distance calculation of distributions.
        """
        # permutate cells order and compute KL div
        n_cell = self.data.x.shape[1]
        dis_p = []
        np.random.seed(0)
        for _ in tqdm(range(n), desc="Permutating", total = n):
            # p: pmt, v: virtual, m: mean, S: sigma
            idx_pmt = np.random.choice(
                np.arange(n_cell), size=n_cell
            )  # bootstrap cell labels
            data_WT_p = Data(
                # x=torch.tensor(data.x[:, idx_pmt], dtype=torch.float),
                x=self.data.x[:, idx_pmt].clone().detach().requires_grad_(False),
                edge_index=self.data.edge_index,
            )
            data_KO_p = Data(
                # x=torch.tensor(data_v.x[:, idx_pmt], dtype=torch.float),
                x=data_v.x[:, idx_pmt].clone().detach().requires_grad_(False),
                edge_index=data_v.edge_index,
            )  # construct virtual data (KO) based on pmt WT
            z_mp, z_Sp = self.get_latent_vars(data_WT_p)
            z_mvp, z_Svp = self.get_latent_vars(data_KO_p)
            if by == "KL":
                dis_p.append(get_distance(z_mvp, z_Svp, z_mp, z_Sp))  # order KL
            # if by == 'reverse_KL':
            #     dis_p.append(get_distance(z_mp, z_Sp, z_mvp, z_Svp)) # reverse order KL
            if by == "t":
                dis_p.append(get_distance(z_mvp, z_Svp, z_mp, z_Sp, by="t"))
            if by == "EMD":
                dis_p.append(get_distance(z_mvp, z_Svp, z_mp, z_Sp, by="EMD"))
        return np.array(dis_p)


def eva(args):
    import pickle
    current_dir =  os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    with open(os.path.join(current_dir, "data.p"), 'rb') as f, open(os.path.join(current_dir, "data_ko.p"), 'rb') as f_ko:
        data = pickle.load(f)
        data_ko = pickle.load(f_ko)
    data, data_ko = Data.from_dict(data), Data.from_dict(data_ko)
    hyperparams = {"epochs": 100, 
                    "lr": 7e-4, 
                    "beta": 1e-4, 
                    "seed": None}
    sensei = VGAE_trainer(data, 
                         epochs=hyperparams["epochs"], 
                         lr=hyperparams["lr"], 
                         log_dir=args.logdir, 
                         beta=hyperparams["beta"],
                         verbose = args.verbose,
                         )
    sensei.train(edge_noise = args.e_noise, 
                 x_noise = args.x_noise)
    epoch, loss, auc, ap = sensei.final_metrics
    try:
        f = open(os.path.join(current_dir, f'{args.train_out}.txt'), 'r')
    except IOError:
        f = open(os.path.join(current_dir, f'{args.train_out}.txt'), 'w')
        f.writelines("Epoch,Loss,AUROC,AP\n")
    finally:  
        f = open(os.path.join(current_dir, f'{args.train_out}.txt'), 'a')
        f.writelines(f"{epoch:03d}, {loss:.4f}, {auc:.4f}, {ap:.4f}\n")					
        f.close()
    if args.do_test:
        print("continue")
        from .utils import get_generank
        from GenKI.utils import get_r2_score
        z_mu, z_std = sensei.get_latent_vars(data)
        z_mu_KO, z_std_KO = sensei.get_latent_vars(data_ko)
        dis = get_distance(z_mu_KO, z_std_KO, z_mu, z_std, by = 'KL')
        null = sensei.pmt(data_ko, n = 100, by = 'KL')
        res = get_generank(data, dis, null, save_significant_as = args.generank_out)
        geneset = list(res.index)
        r2, r2_adj = get_r2_score(data, geneset[1:], geneset[0])
        try:
            f = open(os.path.join(current_dir, f'{args.r2_out}.txt'), 'r')
        except IOError:
            f = open(os.path.join(current_dir, f'{args.r2_out}.txt'), 'w')
            f.writelines("R2,R2_adj\n")
        finally:  
            f = open(os.path.join(current_dir, f'{args.r2_out}.txt'), 'a')
            f.writelines(f"{r2:.4f}, {r2_adj:.4f}\n")					
            f.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('file', type = str)
    parser.add_argument('-l', '--logdir', type = str, default = None)
    parser.add_argument('-e', '--e_noise', type = float, default = None)
    parser.add_argument('-x', '--x_noise', type = float, default = None)
    parser.add_argument('-v', '--verbose', action = 'store_true')
    parser.add_argument('--train_out', type = str, default = 'sum_train')
    parser.add_argument('--generank_out', type = str, default = 'gene_list')
    parser.add_argument('--r2_out', type = str, default = 'gene_list')
    parser.add_argument('--do_test', action = 'store_true')
    args = parser.parse_args()

    eva(args)

    # python -m GenKI.train -v --train_out train_sum --do_test --generank_out genelist --r2_out r2_score


