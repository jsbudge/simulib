from glob import glob
from typing import List, Optional, Union
from pytorch_lightning import LightningDataModule
from skimage.feature import multiscale_basic_features
from skimage.filters import gabor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as tv2
import torch
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm
from PIL import Image


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for _ in np.arange(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout


def savePNGtoData(path, fname, label, data):
    total_data = []
    total_labels = []
    for x in tqdm(range(0, data.shape[0] - 256, 128)):
        for y in range(0, data.shape[1] - 256, 128):
            rl = label[x:x + 256, y:y + 256, :] + 0
            relabel = label[x:x + 256, y:y + 256, :] + 0
            relabel[:, :, 0] = np.logical_and(np.logical_and(rl[:, :, 0] == 255, rl[:, :, 1] == 0), rl[:, :, 2] == 0)
            relabel[:, :, 1] = np.logical_and(np.logical_and(rl[:, :, 1] == 255, rl[:, :, 0] == 0), rl[:, :, 2] == 0)
            relabel[:, :, 2] = np.logical_and(np.logical_and(rl[:, :, 2] == 255, rl[:, :, 1] == 0), rl[:, :, 0] == 0)
            total_data.append(data[x:x + 256, y:y + 256])
            total_labels.append(relabel)
    total_data = np.stack(total_data).astype(np.float32)
    total_labels = np.stack(total_labels).astype(np.float32).swapaxes(1, 3).swapaxes(2, 3)
    try:
        total_data.tofile(f'{path}/base_{fname}.dat')
        total_labels.tofile(f'{path}/labeled_{fname}.dat')
    except Exception:
        return False
    return True


class RCSDataset(Dataset):
    def __init__(self, datapath='./data', split=1000):
        # Load in data
        data_files = glob(f'{datapath}/base_*.png')
        data_list = []
        label_list = []
        for d in data_files:
            data = np.array(Image.open(d).convert('L')) / 255.
            data = anisodiff(data, 5, gamma=.25, kappa=1000)
            aug_data = np.zeros((data.shape[0], data.shape[1], 4))
            aug_data[:, :, 0] = data + 0.
            aug_data[:, :, 1] = data + 0.
            aug_data[:, :, 2] = data + 0.
            aug_data[:, :, 3] = data + 0.

            data_list.append(aug_data)
            labels = np.array(Image.open(d.replace('base', 'labeled')))
            relabel = np.zeros((labels.shape[0], labels.shape[1], 4))
            relabel[:, :, 0] = np.logical_and(np.logical_and(labels[:, :, 0] == 255, labels[:, :, 1] == 0),
                                              labels[:, :, 2] == 0)
            relabel[:, :, 1] = np.logical_and(np.logical_and(labels[:, :, 1] == 255, labels[:, :, 0] == 0),
                                              labels[:, :, 2] == 0)
            relabel[:, :, 2] = np.logical_and(np.logical_and(labels[:, :, 2] == 255, labels[:, :, 1] == 0),
                                              labels[:, :, 0] == 0)
            relabel[:, :, 3] = np.logical_not(np.any(relabel[:, :, :3], axis=2))
            label_list.append(relabel)
        self.data = torch.tensor(np.stack(data_list).swapaxes(1, 3).swapaxes(2, 3), dtype=torch.float32)
        self.labels = torch.tensor(np.stack(label_list).swapaxes(1, 3).swapaxes(2, 3), dtype=torch.float32)
        self.cat_data = torch.cat([self.data.unsqueeze(0), self.labels.unsqueeze(0)], 0)
        self.transform = tv2.Compose([tv2.RandomCrop(size=(256, 256)),
                                      tv2.RandomVerticalFlip(),
                                      tv2.RandomHorizontalFlip()])
        self.sz = split

    def __getitem__(self, idx):
        fidx = idx % self.cat_data.shape[1]
        it = self.transform(self.cat_data[:, fidx, :, :, :])
        return it[0][0, :, :].unsqueeze(0), it[1]

    def __len__(self):
        return self.sz


def collate_fun(batch):
    return (torch.stack([ccd for ccd, _, _, _, _, _ in batch]), torch.stack([tcd for _, tcd, _, _, _, _ in batch]),
            torch.stack([csd for _, _, csd, _, _, _ in batch]), torch.stack([tsd for _, _, _, tsd, _, _ in batch]),
            torch.tensor([pl for _, _, _, _, pl, _ in batch]), torch.tensor([bw for _, _, _, _, _, bw in batch]))


class BaseModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            collate: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = 0 #cpu_count() // 2
        self.pin_memory = pin_memory
        self.single_example = single_example
        self.device = device
        self.collate = collate

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        if self.collate:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
                collate_fn=collate_fun,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.collate:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory,
                collate_fn=collate_fun,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory,
            )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fun,
        )


class RCSModule(BaseModule):
    def __init__(
            self,
            dataset_size: int = 256,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            split: float = .7,
            data_patches: int = 1000,
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.dataset_size = dataset_size
        self.train_split = int(split * data_patches)
        self.val_split = int((1 - split) * data_patches)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = RCSDataset(split=self.train_split)
        self.val_dataset = RCSDataset(split=self.val_split)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = np.array(Image.open('/home/jeff/repo/simulib/data/base_SAR_06212024_124710.png').convert('L')) / 255.
    aug_data = multiscale_basic_features(data, intensity=False, num_sigma=1)
    labels = np.array(Image.open('/home/jeff/repo/simulib/data/labeled_SAR_06212024_124710.png'))
    savePNGtoData('./data', 'SAR_06212024_124710', labels, data)

    test = np.fromfile('/home/jeff/repo/simulib/data/labeled_SAR_06212024_124710.dat', np.float32).reshape(
        (-1, 3, 256, 256))
    plt.figure()
    plt.imshow(test[100, 1, ...])
    plt.show()