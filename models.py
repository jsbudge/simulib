import contextlib
import itertools
from typing import List, Any
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as tf
from abc import abstractmethod
from pytorch_lightning import LightningModule
import numpy as np
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward


def calc_conv_size(inp_sz, kernel_sz, stride, padding):
    return np.floor((inp_sz - kernel_sz + 2 * padding) / stride) + 1


def init_weights(m):
    with contextlib.suppress(ValueError):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)
        # sourcery skip: merge-nested-ifs
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(.01)


class GaborBank(LightningModule):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        freqs = [.1, .3, .5, .7]
        orients = np.linspace(0, 2 * np.pi, 17)[:-1]
        kernels = np.zeros((len(freqs) * len(orients), 1, 35, 35))
        for idx, (f, o) in enumerate(itertools.product(freqs, orients)):
            ker = gabor_kernel(f, o).real
            kernels[idx, 0, 17 - ker.shape[0] // 2:17 + ker.shape[0] // 2 + 1,
            17 - ker.shape[1] // 2:17 + ker.shape[1] // 2 + 1] = ker
        self.kernel = nn.Conv2d(1, kernels.shape[0], 35, 1, 17)
        self.kernel.weight = nn.Parameter(torch.tensor(kernels, dtype=torch.float32), requires_grad=False)
        self.nchan = kernels.shape[0]

    def forward(self, x):
        # x = self.kernel(x)
        # std, mu = torch.std_mean(x, dim=(2, 3))
        # return (x - mu[:, :, None, None]) / std[:, :, None, None]
        return self.kernel(x)


class LKA(LightningModule):

    def __init__(self, channel_sz, kernel_sizes=(3, 3), dilation=6, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        padding = int(dilation * (kernel_sizes[1] - 1) // 2)
        self.kernel = nn.Sequential(
            nn.Conv2d(channel_sz, channel_sz, kernel_sizes[0], 1, int(kernel_sizes[0] // 2)),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, kernel_sizes[1], 1, padding, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 1, 1, 0),
            nn.GELU(),
        )

    def forward(self, x):
        return self.kernel(x) * x


class LKATranspose(LightningModule):

    def __init__(self, channel_sz, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.kernel = nn.Sequential(
            nn.ConvTranspose2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, 3, 1, 6, dilation=6),
            nn.GELU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, 1, 1, 0),
            nn.GELU(),
        )

    def forward(self, x):
        return self.kernel(x) * x


class CRF(LightningModule):
    """
    Class for learning and inference in conditional random field model using mean field approximation
    and convolutional approximation in pairwise potentials term.

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions of input tensors.
    filter_size : int or sequence of ints
        Size of the gaussian filters in message passing.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    n_iter : int
        Number of iterations in mean field approximation.
    requires_grad : bool
        Whether or not to train CRF's parameters.
    returns : str
        Can be 'logits', 'proba', 'log-proba'.
    smoothness_weight : float
        Initial weight of smoothness kernel.
    smoothness_theta : float or sequence of floats
        Initial bandwidths for each spatial feature in the gaussian smoothness kernel.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    """

    def __init__(self, n_spatial_dims, filter_size=11, n_iter=5, requires_grad=True,
                 returns='logits', smoothness_weight=10, smoothness_theta=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.returns = returns
        self.requires_grad = requires_grad

        self._set_param('smoothness_weight', smoothness_weight)
        self._set_param('inv_smoothness_theta', 1 / np.broadcast_to(smoothness_theta, n_spatial_dims))

    def _set_param(self, name, init_value):
        setattr(self, name, nn.Parameter(torch.tensor(init_value, dtype=torch.float, requires_grad=self.requires_grad)))

    def forward(self, x, spatial_spacings=None, verbose=False):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. the CNN's output.
        spatial_spacings : array of floats or None
            Array of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.
            None is equivalent to all ones. Used to adapt spatial gaussian filters to different inputs' resolutions.
        verbose : bool
            Whether to display the iterations using tqdm-bar.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``
            with logits or (log-)probabilities of assignment to each class.
        """
        batch_size, n_classes, *spatial = x.shape
        assert len(spatial) == self.n_spatial_dims

        # binary segmentation case
        if n_classes == 1:
            x = torch.cat([x, torch.zeros(x.shape).to(x)], dim=1)

        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, self.n_spatial_dims))

        negative_unary = x.clone()

        for _ in range(self.n_iter):
            # normalizing
            x = tf.softmax(x, dim=1)

            # message passing
            x = self.smoothness_weight * self._smoothing_filter(x, spatial_spacings)

            # compatibility transform
            x = self._compatibility_transform(x)

            # adding unary potentials
            x = negative_unary - x

        if self.returns == 'logits':
            output = x
        elif self.returns == 'proba':
            output = tf.softmax(x, dim=1)
        elif self.returns == 'log-proba':
            output = tf.log_softmax(x, dim=1)
        else:
            raise ValueError("Attribute ``returns`` must be 'logits', 'proba' or 'log-proba'.")

        if n_classes == 1:
            output = output[:, 0] - output[:, 1] if self.returns == 'logits' else output[:, 0]
            output.unsqueeze_(1)

        return output

    def _smoothing_filter(self, x, spatial_spacings):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        return torch.stack([self._single_smoothing_filter(x[i], spatial_spacings[i]) for i in range(x.shape[0])])

    @staticmethod
    def _pad(x, filter_size):
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]

        return tf.pad(x, list(reversed(padding)))  # F.pad pads from the end

    def _single_smoothing_filter(self, x, spatial_spacing):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        spatial_spacing : sequence of len(spatial) floats

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        """
        x = self._pad(x, self.filter_size)
        for i, dim in enumerate(range(1, x.ndim)):
            # reshape to (-1, 1, x.shape[dim])
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)

            # 1d gaussian filtering
            kernel = self._create_gaussian_kernel1d(self.inv_smoothness_theta[i], spatial_spacing[i],
                                                    self.filter_size[i]).view(1, 1, -1).to(x)
            x = tf.conv1d(x, kernel)

            # reshape back to (n, *spatial)
            x = x.squeeze(1).view(*shape_before_flatten, x.shape[-1]).transpose(-1, dim)

        return x

    @staticmethod
    def _create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        """
        Parameters
        ----------
        inverse_theta : torch.tensor
            Tensor of shape ``(,)``
        spacing : float
        filter_size : int

        Returns
        -------
        kernel : torch.tensor
            Tensor of shape ``(filter_size,)``.
        """
        distances = spacing * torch.arange(-(filter_size // 2), filter_size // 2 + 1).to(inverse_theta)
        kernel = torch.exp(-(distances * inverse_theta) ** 2 / 2)
        zero_center = torch.ones(filter_size).to(kernel)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape ``(batch_size, n_classes, *spatial)``.

        Returns
        -------
        output : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        labels = torch.arange(x.shape[1])
        compatibility_matrix = self._compatibility_function(labels, labels.unsqueeze(1)).to(x)
        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @staticmethod
    def _compatibility_function(label1, label2):
        """
        Input tensors must be broadcastable.

        Parameters
        ----------
        label1 : torch.Tensor
        label2 : torch.Tensor

        Returns
        -------
        compatibility : torch.Tensor
        """
        return -(label1 == label2).float()


class FlatModule(LightningModule):

    def __init__(self):
        super(FlatModule, self).__init__()

    def get_flat_params(self):
        """Get flattened and concatenated params of the model."""
        return torch.cat([torch.flatten(p) for _, p in self._get_params().items()])

    def _get_params(self):
        return {name: param.data for name, param in self.named_parameters()}

    def init_from_flat_params(self, flat_params):
        """Set all model parameters from the flattened form."""
        assert isinstance(flat_params, torch.Tensor), "Argument to init_from_flat_params() must be torch.Tensor"
        state_dict = self._unflatten_to_state_dict(flat_params, self._get_param_shapes())
        for name, params in self.state_dict().items():
            if name not in state_dict:
                state_dict[name] = params
        self.load_state_dict(state_dict, strict=True)

    def _unflatten_to_state_dict(self, flat_w, shapes):
        state_dict = {}
        counter = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = flat_w[counter: counter + tnum].reshape(tsize)
            state_dict[name] = torch.nn.Parameter(param)
            counter += tnum
        assert counter == len(flat_w), "counter must reach the end of weight vector"
        return state_dict

    def _get_param_shapes(self):
        return [
            (name, param.shape, param.numel())
            for name, param in self.named_parameters()
        ]


class ImageSegmenter(FlatModule):
    def __init__(self,
                 in_channels: int,
                 label_sz: int,
                 params: dict,
                 channel_sz: int = 32,
                 **kwargs) -> None:
        super(ImageSegmenter, self).__init__()

        self.params = params
        self.channel_sz = channel_sz
        self.in_channels = in_channels
        self.automatic_optimization = False
        self.label_sz = label_sz
        out_sz = 512
        self.feedthrough = nn.Sequential(
            nn.Conv2d(in_channels, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 4, 2, 1),
            nn.GELU(),
        )
        self.wavelet_0 = nn.Sequential(
            nn.Conv2d(3, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 0),
            nn.GELU(),
        )
        self.wavelet_1 = nn.Sequential(
            nn.Conv2d(3, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 4, 1, 0),
            nn.GELU(),
        )
        self.wavelet_2 = nn.Sequential(
            nn.Conv2d(3, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 5, 1, 0),
            nn.GELU(),
        )
        self.fusion_0 = nn.Sequential(
            nn.LayerNorm(out_sz // 2),
            LKA(channel_sz, kernel_sizes=(5, 3), dilation=9),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 4, 2, 1),
            nn.GELU(),
        )
        self.fusion_1 = nn.Sequential(
            nn.LayerNorm(out_sz // 4),
            LKA(channel_sz, kernel_sizes=(5, 3), dilation=9),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 4, 2, 1),
            nn.GELU(),
        )
        self.fusion_2 = nn.Sequential(
            nn.LayerNorm(out_sz // 8),
            LKA(channel_sz, kernel_sizes=(5, 3), dilation=9),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
        )
        self.inflate = nn.Sequential(
            nn.ConvTranspose2d(channel_sz, channel_sz, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, 4, 2, 1),
            nn.GELU(),
            nn.LayerNorm(out_sz),
        )
        self.feedfinal = nn.Sequential(
            LKA(channel_sz, kernel_sizes=(5, 3), dilation=9),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm(out_sz),
            nn.Conv2d(channel_sz, label_sz, 1, 1, 0),
            nn.Softmax2d(),
        )

        self.crf = CRF(2, returns='proba', smoothness_weight=15., smoothness_theta=1.)

        self.dwt = DWTForward(J=3, mode='zero', wave='db3')

        self.out_sz = out_sz
        self.example_input_array = torch.randn((1, in_channels, out_sz, out_sz))
        self.loss_weight = torch.tensor([.9, .9, .9, .01, .01], dtype=torch.float32)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        Yl, Yh = self.dwt(inp)
        inp = self.fusion_0(self.feedthrough(inp) + self.wavelet_0(Yh[0].squeeze(1)))
        inp = self.fusion_1(inp + self.wavelet_1(Yh[1].squeeze(1)))
        inp = self.fusion_2(inp + self.wavelet_2(Yh[2].squeeze(1)))
        inp = self.inflate(inp)
        inp = self.feedfinal(inp)
        return self.crf(inp)

    def loss_function(self, y, y_pred):
        # overlap = torch.sum(torch.argmax(y, dim=1) == torch.argmax(y_pred, dim=1))
        # return 1 - (overlap / (torch.sum(y_pred == 1) + torch.sum(y >= .25) - overlap)) / y.shape[0]
        loss = 0.
        for idx, l in enumerate(self.loss_weight):
            loss += torch.nanmean((y[:, idx, :, :] - y_pred[:, idx, :, :])**2) * l
        return loss

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_after_backward(self) -> None:
        if self.trainer.is_global_zero and self.global_step % 100 == 0 and self.logger:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.global_step)

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning']:
            torch.save(self.state_dict(), './model/inference_model.state')
            print('Model saved to disk.')

            if self.current_epoch % 5 == 0:
                pass

    def on_train_epoch_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['loss_landscape']:
            self.optim_path.append(self.model.get_flat_params())

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
            self.log('LR', sch.get_last_lr()[0], rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)
        scheds = [scheduler]

        return optims, scheds

    def train_val_get(self, batch, batch_idx, kind='train'):
        data, label = batch

        results = self.forward(data)
        train_loss = self.loss_function(results, label)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss
