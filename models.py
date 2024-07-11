import contextlib
from typing import List, Any
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as tf
from abc import abstractmethod
from pytorch_lightning import LightningModule
import numpy as np
import matplotlib.pyplot as plt


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
                 params: dict,
                 channel_sz: int = 32,
                 **kwargs) -> None:
        super(ImageSegmenter, self).__init__()

        self.params = params
        self.channel_sz = channel_sz
        self.in_channels = in_channels
        self.automatic_optimization = False
        out_sz = 256

        # Encoder
        self.channel_conv = nn.Conv2d(in_channels, channel_sz, 1, 1, 0)
        self.big_features = nn.Sequential(
            nn.Conv2d(channel_sz, channel_sz, 15, 1, 7),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 15, 1, 7),
            nn.GELU(),
            nn.LayerNorm(out_sz),
        )
        self.medium_features = nn.Sequential(
            nn.Conv2d(channel_sz, channel_sz, 7, 1, 3),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 7, 1, 3),
            nn.GELU(),
            nn.LayerNorm(out_sz),
        )
        self.little_features = nn.Sequential(
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm(out_sz),
        )
        self.feed_stack = nn.ModuleList()
        for _ in range(3):
            self.feed_stack.append(nn.Sequential(
                nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channel_sz, channel_sz, 3, 1, 6, dilation=6),
                nn.GELU(),
                nn.Conv2d(channel_sz, channel_sz, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channel_sz, channel_sz, 3, 1, 6, dilation=6),
                nn.GELU(),
                nn.Conv2d(channel_sz, channel_sz, 1, 1, 0),
                nn.GELU(),
                nn.LayerNorm(out_sz),
            ))
        self.feedfinal = nn.Sequential(
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm(out_sz),
            nn.Conv2d(channel_sz, 4, 3, 1, 1),
            nn.Softmax2d(),
        )

        self.out_sz = out_sz
        self.example_input_array = torch.randn((1, in_channels, 256, 256))

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        inp = self.channel_conv(inp)
        z = self.feed_stack[0](inp) + self.big_features(inp)
        z = self.feed_stack[1](z) + self.medium_features(inp)
        z = self.feed_stack[2](z) + self.little_features(inp)
        return self.feedfinal(z)

    def loss_function(self, y, y_pred):
        return tf.mse_loss(y, y_pred)

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