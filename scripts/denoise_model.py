import numpy as np
import torch.nn as nn
import layers as B
import torch

"""
# --------------------------------------------
# FFDNet (15 or 12 conv layers)
# --------------------------------------------
Reference:
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
"""


# --------------------------------------------
# FFDNet
# --------------------------------------------
class FFDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = B.PixelUnShuffle(upscale_factor=sf)

        m_head = B.conv(in_nc*sf*sf+1, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc*sf*sf, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        if sigma.size()[2]==1:
            m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        else:
            m = self.m_down(sigma)[:,0,:,:].unsqueeze(axis=1)
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)

        x = x[..., :h, :w]
        return x


if __name__ == '__main__':
    pix_data = np.random.rand(240, 240, 1)
    model_path = './model_zoo/ffdnet_gray.pth'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    img_L = np.float32(pix_data / pix_data.max())

    img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0)
    img_L = img_L.to(device)

    sigma = torch.full((1, 1, 1, 1), 2 / 255.).type_as(img_L)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------

    img_E = model(img_L, sigma)
    img_E = img_E.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    img_E = np.uint8((img_E*255.0).round())

    #  run models/network_ffdnet.py
