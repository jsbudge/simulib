import numpy as np
from glob import glob
from SDRParsing import loadASIFile, loadASHFile
from simulation_functions import db
from tqdm import tqdm

root_dir = '/data6/SAR_DATA/2023'
max_val = 229.3302
max_north = 40.147821
min_north = 40.129011
max_east = -111.649584
min_east = -111.673924
pairs = {a: f'{a[:-1]}i' for a in glob(f'{root_dir}/*/SAR_*.ash', root_dir=root_dir, recursive=True)}

for _ash, _asi in tqdm(pairs.items()):
    ash = loadASHFile(_ash)
    asi = db(loadASIFile(_asi, nrows=ash['image']['nRows'], ncols=ash['image']['nCols']))
    asi = asi[asi > -300]
    if len(asi) > 0:
        max_val = max(max_val, asi.max())
        if min_north < ash['geo']['north'] < max_north or min_north < ash['geo']['south'] < max_north:
            if min_east < ash['geo']['east'] < max_east or min_east < ash['geo']['west'] < max_east:
                pass

