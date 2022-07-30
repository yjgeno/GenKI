import numpy as np
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans2
import math


def boxcox_norm(x):
    """
    x: 1-D array-like, require positive values.
    Box-cox transform and standarize x
    """
    xt, _ = scipy.stats.boxcox(x)
    return StandardScaler().fit_transform(xt[:, None]).flatten()
    # (xt - xt.mean())/np.sqrt(xt.var()) # z-score for 1-D


def _t_stat(m0, S0, m1, S1):
    return (m0 - m1) / math.sqrt(
        S0**2 + S1**2
    )  # / math.log(S1/S0) #math.sqrt(S0**2 + S1**2)


def _kl_1d(m0, S0, m1, S1):
    """
    KL divergence between two gaussian distributions.
    https://stats.stackexchange.com/questions/234757/how-to-use-kullback-leibler-divergence-if-mean-and-standard-deviation-of-of-two
    """
    return 0.5 * math.log(S1 / S0) + (S0**2 + (m0 - m1) ** 2) / (2 * S1**2) - 1 / 2


def _kl_mvn(m0, S0, m1, S1):
    """
    KL divergence between two multivariate gaussian distributions.
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.pinv(S1)  # pseudo-inverse
    diff = m1 - m0
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(
        np.linalg.det(S1) / np.linalg.det(S0)
    )  # np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ iS1 @ diff  # np.sum( (diff*diff) * iS1, axis=1)

    return 0.5 * (tr_term + det_term + quad_term - N)


def _wasserstein_dist2(m0, S0, m1, S1):
    """
    Earth Mover Distance (wasserstein distance) between two multivariate gaussian distributions.
    """
    return np.square(np.linalg.norm(m0 - m1)) + np.trace(
        S0 + S1 - 2 * np.sqrt(np.sqrt(S0) @ S1 @ np.sqrt(S0))
    )


def get_distance(z_m0, z_S0, z_m1, z_S1, by = "KL"):
    """
    z_m0, z_S0: latent means and sigma from WT data.
    z_mv, z_Sv: latent means and sigma from virtual data.
    """
    dis = []
    try:
        out_channels = z_m0.shape[1]  # out_channels >= 2
        for m0, S0, m1, S1 in zip(z_m0, z_S0, z_m1, z_S1):
            temp_S0 = np.zeros((out_channels, out_channels), float)
            np.fill_diagonal(temp_S0, S0)
            temp_S1 = np.zeros((out_channels, out_channels), float)
            np.fill_diagonal(temp_S1, S1)
            if by == "KL":
                dis.append(_kl_mvn(m0, temp_S0, m1, temp_S1))
                # dis.append(
                #     (
                #         _kl_mvn(m0, temp_S0, m1, temp_S1)
                #         + _kl_mvn(m1, temp_S1, m0, temp_S0)
                #     )
                #     / 2
                # )
            if by == "EMD":
                dis.append(_wasserstein_dist2(m0, temp_S0, m1, temp_S1))
    except IndexError:  # when out_channels == 1
        for m0, S0, m1, S1 in zip(z_m0, z_S0, z_m1, z_S1):
            if by == "KL":
                dis.append((_kl_1d(m0, S0, m1, S1) + _kl_1d(m1, S1, m0, S0)) / 2)
            if by == "t":
                dis.append(_t_stat(m0, S0, m1, S1))
    return np.array(dis)


def get_generank(
    data,
    distance,
    null = None,
    rank: bool = True,
    reverse: bool = False,
    bagging: float = 0.05,
    cutoff: float = 0.95,
    save_significant_as: bool = None,
):
    """
    data: torch_geometric.data.data.Data.
    distance: array-like, output of get_distance.
    null: array-like, output of pmt.
    bagging: threshold for bagging top names at each permutation.
    cutoff: threshold for frequency of bagging after all permutations.
    save_significant_as: .txt file name of significant genes that will be saved in result for enrichment test.
    """
    if null is not None:
        if reverse:
            idx = np.argsort(-null, axis=1)
        else:
            idx = np.argsort(null, axis=1)
        thre = int(len(data.y) * (1 - bagging))  # 95%
        idx = idx[:, thre:].flatten()  # bagging index of top ranked genes

        y = np.bincount(idx)
        ii = np.nonzero(y)[0]
        freq = np.vstack((ii, y[ii])).T  # [gene_index, #hit]
        df_KL = pd.DataFrame(
            data=distance[freq[:, 0]][:, None],
            index=np.array(data.y)[freq[:, 0]],
            columns=["dis"],
        )
        df_KL[["index", "hit"]] = freq.astype(int)
        hit = int(null.shape[0] * cutoff)
        df_KL = df_KL[(df_KL.hit >= hit) & (df_KL.dis != 0)]
        if rank:
            df_KL.sort_values(by=["dis", "hit"], ascending=reverse, inplace=True)
        if save_significant_as is not None:
            output = list(df_KL.index)
            os.makedirs("result", exist_ok=True)
            np.savetxt(
                os.path.join("result", f"{save_significant_as}.txt"), output, fmt="%s", delimiter=","
            )
            print(f"save {len(output)} genes to \"./result/{save_significant_as}.txt\"")
    else:
        df_KL = pd.DataFrame(
            data=distance, index=np.array(data.y), columns=["dis"]
        )  # filter pseudo values
        if rank:
            df_KL.sort_values(by=["dis"], ascending=reverse, inplace=True)
    if rank:
        df_KL["rank"] = np.arange(len(df_KL)) + 1
    return df_KL


def get_generank_gsea(data, distance, reverse=False, save_as=None):
    """
    data: torch_geometric.data.data.Data, 
    distance: array-like, output of get_distance.
    save_as: str, .txt file name that will be saved in result for GSEA.
    """
    df_gsea = get_generank(data, distance, reverse=reverse)
    df_gsea = df_gsea[df_gsea["dis"] > 0]  # remove pinverse

    # Box-cox
    df_gsea["dis_norm"] = boxcox_norm(df_gsea["dis"])  # z-score

    df_gsea.sort_values(by="dis_norm", inplace=True, ascending=reverse)
    output_gsea = np.stack(
        (df_gsea.index, df_gsea["dis_norm"])
    ).T  # GSEA: [gene_name, value]

    if save_as is not None:
        os.makedirs("./result", exist_ok=True)
        np.savetxt(
            f"./result/GSEA_{save_as}.txt", output_gsea, fmt="%s", delimiter="\t"
        )
        print(f"save ranked genes to \"./result/GSEA_{save_as}.txt\"")
    return df_gsea


def get_sys_KO_cluster(
    obj,
    sys_res: np.ndarray,
    perplexity=25,
    n_cluster=50,
    save_as=None,
    show_TSNE=True,
    verbose=False,
):
    """
    obj: a sc object.
    sys_res: np.ndarray, output of run_sys_KO.
    perplexity: int, hyperparameter of TSNE.
    n_cluster: int, hyperparameter of k-means.
    show_TSNE: bool, whether to show the TSNE plot after clustering.
    """
    np.random.seed(100)
    scaled_sys_res = StandardScaler().fit_transform(
        sys_res
    )  # standarize features (cols)
    X_embedded = TSNE(
        n_components=2,
        learning_rate="auto",
        perplexity=perplexity,
        metric="euclidean",
        init="pca",
    ).fit_transform(scaled_sys_res)
    _, label = kmeans2(X_embedded, n_cluster, minit="points")  # centroid
    cluster_idx = np.where(label == label[obj(obj._target_gene)])
    cluster_gene_names = np.array(obj._gene_names)[cluster_idx]
    if verbose:
        print(
            f"TSNE perplexity: {perplexity}, # Clustering: {n_cluster}\nThe cluster containing {obj._target_gene} has {len(cluster_gene_names)} genes"
        )
    if save_as is not None:
        os.makedirs("result", exist_ok=True)
        np.savetxt(
            os.path.join("result", f"sys_KO_{save_as}.txt"),
            cluster_gene_names,
            fmt="%s",
            delimiter="\t",
        )
        print(f"save ranked genes to \"./result/sys_KO_{save_as}.txt\"")
    if show_TSNE:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
        colors = [
            "red" if i == label[obj(obj._target_gene)] else "black" for i in label
        ]
        ax.set_title(f"TSNE plot of systematic KO {len(sys_res)} genes")
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=3, c=colors, alpha=0.5)
        ax.axis("tight")
        ax.annotate(
            obj._target_gene,
            xy=(X_embedded[obj(obj._target_gene)]),
            xycoords="data",
            xytext=(X_embedded[obj(obj._target_gene)] + 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        plt.show()
    return cluster_gene_names


if __name__ == "__main__":
    pm = np.array([0, 0])
    pv = np.array([0, 0])
    qm = np.array([[1, 0], [0, 1]])
    qv = np.array([[1, 0], [0, 1]])
    assert _kl_mvn(pm, qm, pv, qv) == 0
