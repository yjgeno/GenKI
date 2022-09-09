import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from .model import VGAE
import torch.utils.tensorboard as tb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import get_distance
from .preprocesing import split_data


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
                 epochs: int = 100,
                 lr: float = 7e-4,
                 weight_decay = 9e-4,
                 beta: str = 1e-4,
                 log_dir: str = None, 
                 verbose: bool = True,            
                 seed: int = None,
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
        self.weight_decay = weight_decay 
        self.seed = seed


    def __repr__(self) -> str:
        return f"Hyperparameters\n"\
               f"epochs: {self.epochs}, lr: {self.lr}, beta: {self.beta:.4f}\n"


    # split data
    def _transform_data(self,
                        x_noise: float = None, 
                        x_dropout: float = None,
                        edge_noise: float = None, 
                        **kwargs
                        ):
        """
        Args:
        x_noise: Standard deviation of white noise added to the training data. Defaults to None.
        edge_noise: Remove or add edges to the training data.
        """
        # from copy import deepcopy
        # data_ = deepcopy(self.data)
        if x_dropout is not None:
            mask = torch.FloatTensor(self.data.x.shape).uniform_() > x_dropout # % zeros
            self.data.x = self.data.x * mask
            print(f"force zeros to data x, dropout: {x_dropout}")

        self.train_data, self.val_data, self.test_data = split_data(data = self.data, **kwargs) # fixed split
        if x_noise is not None: # white noise on X
            gamma = x_noise * torch.randn(self.train_data.x.shape)
            self.train_data.x = 2**gamma * self.train_data.x
            print(f"add white noise to training data x, level: {x_noise} SD")
        
        # if x_dropout is not None:
        #     mask = torch.FloatTensor(self.train_data.x.shape).uniform_() > x_dropout # % zeros
        #     self.train_data.x = self.train_data.x * mask
        #     print(f"force zeros to training data x, dropout: {x_dropout}")
		
        if edge_noise is not None:
            n_pos_edge = self.train_data.pos_edge_label_index.shape[1]
            n = int(abs(edge_noise) * n_pos_edge)
            print("Before:", self.train_data)
            print("\n")
            if edge_noise > 0: # fold edges
                from torch_geometric.utils import negative_sampling
                fake_pos_edge = negative_sampling(self.data.edge_index,  # then fake edges impossible appeared in test set: for data leakage
                                                    num_neg_samples= n,
                                                    num_nodes = len(self.train_data.x))
                new_pos_edge = torch.unique(torch.cat((self.train_data.pos_edge_label_index, fake_pos_edge), 1), dim = 1) 
                new_edge = torch.cat((new_pos_edge, new_pos_edge[torch.LongTensor([1, 0])]), 1) # swap           
                print(f"add noise to training data edge, level: {edge_noise}: add {n} edges")      
            else: # ratio edges
                if n > n_pos_edge:
                    raise ValueError("cannot retain edges more than the total")
                weights = torch.tensor([1/n_pos_edge] * n_pos_edge, dtype = torch.float) # uniform weights
                index = weights.multinomial(num_samples = n, replacement = False)
                new_pos_edge = self.train_data.pos_edge_label_index[:, index]
                new_edge = torch.cat((new_pos_edge, new_pos_edge[torch.LongTensor([1, 0])]), 1)
                # new_neg_edge = self.train_data.neg_edge_label_index[:, index]
                # self.train_data.neg_edge_label_index = new_neg_edge
                print(f"retain a portion of training data edge, level: {abs(edge_noise)}: retain {n} edges")   
            self.train_data.edge_index, self.train_data.pos_edge_label_index = new_edge, new_pos_edge
            print("After:", self.train_data)


    # refer to https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
    def train(self, **kwargs):
        global_step = 0
        if self.seed is not None:
            torch.manual_seed(self.seed) # 8096
        self._transform_data(**kwargs) # get split data w/o noise
        self.model = VGAE(VariationalGCNEncoder(self.num_features, self.out_channels))
        
        # to cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.train_data, self.val_data, self.test_data = self.train_data.to(device), self.val_data.to(device), self.test_data.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
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
    from .preprocesing import load_gdata
    CURRENT_DIR =  os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    load_path = os.path.join(CURRENT_DIR, args.dir)
    data = load_gdata(load_path, "data")   

    sensei = VGAE_trainer(data, 
                         epochs = args.epochs, 
                         lr = args.lr, 
                         log_dir = args.logdir, 
                         beta = args.beta,
                         verbose = args.verbose,
                         seed = args.seed, 
                         )
    sensei.train(edge_noise = args.e_noise, 
                 x_noise = args.x_noise,
                 x_dropout = args.x_dropout,
                 dir = args.dir, load = False, # load split data
                 )
    epoch, loss, auc, ap = sensei.final_metrics

    save_path = os.path.join(CURRENT_DIR, "train_log")
    os.makedirs(save_path, exist_ok = True)
    try:
        f = open(os.path.join(save_path, f'{args.train_out}.txt'), 'r')
    except IOError:
        f = open(os.path.join(save_path, f'{args.train_out}.txt'), 'w')
        f.writelines("Epoch,Loss,AUROC,AP\n")
    finally:  
        f = open(os.path.join(save_path, f'{args.train_out}.txt'), 'a')
        f.writelines(f"{epoch:03d}, {loss:.4f}, {auc:.4f}, {ap:.4f}\n")					
        f.close()
    if args.do_test:
        data_ko = load_gdata(load_path, "data_ko")
        print("continue")
        from .utils import get_generank, get_r2_score
        z_mu, z_std = sensei.get_latent_vars(data)
        z_mu_KO, z_std_KO = sensei.get_latent_vars(data_ko)
        dis = get_distance(z_mu_KO, z_std_KO, z_mu, z_std, by = 'KL')
        null = sensei.pmt(data_ko, n = 100, by = 'KL')
        res = get_generank(data, dis, null, save_significant_as = args.generank_out)
        geneset = list(res.index)
        r2, r2_adj = get_r2_score(data, geneset[1:], geneset[0])
        try:
            f = open(os.path.join(save_path, f'{args.r2_out}.txt'), 'r')
        except IOError:
            f = open(os.path.join(save_path, f'{args.r2_out}.txt'), 'w')
            f.writelines("R2,R2_adj\n")
        finally:  
            f = open(os.path.join(save_path, f'{args.r2_out}.txt'), 'a')
            f.writelines(f"{r2:.4f}, {r2_adj:.4f}\n")					
            f.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddir', type = str, default = "data")
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 7e-4)
    parser.add_argument('--beta', type = float, default = 1e-4)
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--logdir', type = str, default = None)
    parser.add_argument('-E', '--e_noise', type = float, default = None)
    parser.add_argument('-X', '--x_noise', type = float, default = None)
    parser.add_argument('-XO', '--x_dropout', type = float, default = None)
    parser.add_argument('-v', '--verbose', action = "store_true")
    parser.add_argument('--train_out', type = str, default = "train_log")

    parser.add_argument('--do_test', action = "store_true")
    parser.add_argument('--generank_out', type = str, default = "gene_list")
    parser.add_argument('--r2_out', type = str, default = "r2_score")
    args = parser.parse_args()

    eva(args)
    # python -m GenKI.train --dir data --logdir log_dir/run0 --seed 8096 -E -0.1 -v
    # python -m GenKI.train --dir data_covid --logdir log_dir/sigma0_covid --lr 5e-3 --seed 8096 -v
    # python -m GenKI.train --dir data --train_out train_log --do_test --generank_out genelist --r2_out r2_score -v


