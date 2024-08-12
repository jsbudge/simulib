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
            data = np.array(Image.open(d)) / 65535.
            aug_data = np.zeros((data.shape[0], data.shape[1], 5))
            aug_data[:, :, 0] = data + 0.

            data_list.append(aug_data)
            labels = np.array(Image.open(d.replace('base', 'labeled')))
            relabel = np.zeros((labels.shape[0], labels.shape[1], 5))
            # Buildings
            relabel[:, :, 0] = np.logical_and(np.logical_and(labels[:, :, 0] == 255, labels[:, :, 1] == 0),
                                              labels[:, :, 2] == 0)
            # Trees
            relabel[:, :, 1] = np.logical_and(np.logical_and(labels[:, :, 1] == 255, labels[:, :, 0] == 0),
                                              labels[:, :, 2] == 0)
            # Roads
            relabel[:, :, 2] = np.logical_and(np.logical_and(labels[:, :, 2] == 255, labels[:, :, 1] == 0),
                                              labels[:, :, 0] == 0)
            # Fields
            relabel[:, :, 3] = np.logical_and(np.logical_and(labels[:, :, 2] == 255, labels[:, :, 1] == 0),
                                              labels[:, :, 0] == 255)
            # Unlabeled/Unknown
            relabel[:, :, 4] = np.logical_not(np.any(relabel[:, :, :4] > 0, axis=2))
            label_list.append(relabel)
        self.cat_data = [torch.cat(
            [torch.tensor(np.stack([d]).swapaxes(1, 3).swapaxes(2, 3), dtype=torch.float32),
             torch.tensor(np.stack([l]).swapaxes(1, 3).swapaxes(2, 3), dtype=torch.float32)], 0) for d, l in
            zip(data_list, label_list)]
        self.transform = tv2.Compose([tv2.RandomCrop(size=(512, 512)),
                                      tv2.RandomVerticalFlip(),
                                      tv2.RandomHorizontalFlip()])
        self.sz = split

        self.label_sz = label_list[0].shape[2]

    def __getitem__(self, idx):
        fidx = idx % len(self.cat_data)
        it = self.transform(self.cat_data[fidx][:, :, :, :])
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
        self.num_workers = 0  #cpu_count() // 2
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
