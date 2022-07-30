import torch
from torch_geometric.utils import negative_sampling


EPS = 1e-15
MAX_LOGSTD = 10


def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class InnerProductDecoder(torch.nn.Module):
    """The inner product decoder."""
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
    """The Graph Auto-Encoder model."""
    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """Given latent embeddings z, computes the binary cross
        entropy loss for positive edges pos_edge_index and negative
        sampled edges.

        Args:
        z (Tensor): The latent embeddings.
        pos_edge_index (LongTensor): The positive edges to train against.
        neg_edge_index (LongTensor, optional): The negative edges to train
            against. If not given, uses negative sampling to calculate
            negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        """Given latent embeddings z, positive edges
        pos_edge_index and negative edges neg_edge_index
        computes metrics.

        Args:
        z (Tensor): The latent embeddings.
        pos_edge_index (LongTensor): The positive edges to evaluate
            against.
        neg_edge_index (LongTensor): The negative edges to evaluate
            against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score #f1_score, confusion_matrix

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        # pred[pred < 0.5] = 0
        # pred[pred >= 0.5] = 1
        # tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        return roc_auc_score(y, pred), average_precision_score(y, pred) # f1_score(y, pred), [tn, fp, fn, tp]


class VGAE(GAE):
    """The Variational Graph Auto-Encoder model.

    Args:
    encoder (Module): The encoder module to compute :math:`\mu` and
        :math:`\log\sigma^2`.
    decoder (Module, optional): The decoder module. If set to :obj:`None`,
        will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        """Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
        mu (Tensor, optional): The latent space for :math:`\mu`. If set to
            :obj:`None`, uses the last computation of :math:`mu`.
            (default: :obj:`None`)
        logstd (Tensor, optional): The latent space for
            :math:`\log\sigma`.  If set to :obj:`None`, uses the last
            computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
