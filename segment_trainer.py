from glob import glob

from PIL import Image
from clearml import Task
import torch
from torch.nn import functional as tf
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from datamodules import RCSModule
from models import init_weights, ImageSegmenter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# seed_everything(np.random.randint(1, 2048), workers=True)
seed_everything(44, workers=True)

with open('./segmenter_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

# hparams = make_dataclass('hparams', param_dict.items())(**param_dict)

data = RCSModule(**param_dict["dataset_params"])
data.setup()

# Get the model, experiment, logger set up
model = ImageSegmenter(**param_dict['model_params'], label_sz=data.train_dataset.label_sz, params=param_dict)
print('Setting up model...')
tag_warm = 'new_model'
if param_dict['warm_start']:
    print('Model loaded from save state.')
    try:
        model.load_state_dict(torch.load('./model/inference_model.state'))
        tag_warm = 'warm_start'
    except RuntimeError:
        print('Model save file does not match current structure. Re-running with new structure.')
        model.apply(init_weights)
else:
    print('Initializing new model...')
    model.apply(init_weights)

if param_dict['init_task']:
    task = Task.init(project_name='Segmentation', task_name=param_dict['exp_name'])

logger = loggers.TensorBoardLogger(param_dict['log_dir'], name="Segmentation")
expected_lr = max((param_dict['LR'] *
                   param_dict['scheduler_gamma'] ** (param_dict['max_epochs'] *
                                                     param_dict['swa_start'])), 1e-9)
trainer = Trainer(logger=logger, max_epochs=param_dict['max_epochs'],
                  log_every_n_steps=param_dict['log_epoch'], devices=[1], callbacks=
                  [EarlyStopping(monitor='val_loss', patience=param_dict['patience'],
                                 check_finite=True),
                   StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=param_dict['swa_start'])])
# trainer.test(model, train_loader, verbose=True)

print("======= Training =======")
try:
    trainer.fit(model, datamodule=data)
except KeyboardInterrupt:
    if trainer.is_global_zero:
        print('Training interrupted.')
    else:
        print('adios!')
        exit(0)

model.to('cpu')
model.eval()
test_frame = 10
sample, labels = data.val_dataset[test_frame]
recon = model(sample.unsqueeze(0))[0].squeeze(0).data.numpy()
sample = sample.data.numpy()
labels = labels.data.numpy()

if trainer.is_global_zero:
    if param_dict['save_model']:
        try:
            torch.save(model.state_dict(), './model/inference_model.state')
            print('Model saved to disk.')
        except Exception as e:
            print(f'Model not saved: {e}')
    print('Plotting outputs...')
    plt.figure('Original')
    plt.imshow(sample[0])
    plt.figure('Detections')
    plt.axis('off')
    plt.subplot(4, 2, 1)
    plt.title('Buildings')
    plt.imshow(labels[0, ...])
    plt.subplot(4, 2, 2)
    plt.title('Building Detection')
    plt.imshow(recon[0, ...], clim=[0, 1])
    plt.subplot(4, 2, 3)
    plt.title('Trees')
    plt.imshow(labels[1, ...])
    plt.subplot(4, 2, 4)
    plt.title('Tree Detection')
    plt.imshow(recon[1, ...], clim=[0, 1])
    plt.subplot(4, 2, 5)
    plt.title('Roads')
    plt.imshow(labels[2, ...])
    plt.subplot(4, 2, 6)
    plt.title('Road Detection')
    plt.imshow(recon[2, ...], clim=[0, 1])
    plt.subplot(4, 2, 7)
    plt.title('Fields')
    plt.imshow(labels[3, ...])
    plt.subplot(4, 2, 8)
    plt.title('Field Detection')
    plt.imshow(recon[3, ...], clim=[0, 1])

    # Load in the whole deal
    print('Loading in test image for segmentation...')
    model.to('cuda:1')
    idata = np.array(Image.open('/home/jeff/repo/simulib/data/base_SAR_07082024_112333.png')) / 65535.
    segment_image = np.zeros((*idata.shape, 3), dtype=np.uint8)
    colors = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 0, 0)])
    seg_size = 512
    for x in range(0, segment_image.shape[0], seg_size):
        for y in range(0, segment_image.shape[1], seg_size):
            tense = torch.tensor(idata[x:x + seg_size, y:y + seg_size], dtype=torch.float32, device='cuda:1')
            if tense.shape[0] < seg_size or tense.shape[1] < seg_size:
                tense = tf.pad(tense, (0, seg_size - tense.shape[1], 0, seg_size - tense.shape[0]))
            sdata = model(tense.unsqueeze(0).unsqueeze(0))[0].squeeze(0).cpu().data.numpy()
            sdata_arg = np.argmax(sdata, axis=0)
            cdata = np.zeros((seg_size, seg_size, 3), dtype=np.uint8)
            for seg in range(sdata.shape[0]):
                cdata[sdata_arg == seg, :] = colors[seg][None, :]
            segment_image[x:x + min(seg_size, segment_image.shape[0] - x),
            y:y + min(seg_size, segment_image.shape[1] - y), :] = cdata[:min(seg_size, segment_image.shape[0] - x),
                                                             :min(seg_size, segment_image.shape[1] - y), :]
    plt.figure('Segmented Image')
    plt.imshow(idata, cmap='gray')
    plt.imshow(segment_image, alpha=.5)
    plt.axis('tight')
    plt.show(block=True)
    if param_dict['init_task']:
        task.close()
