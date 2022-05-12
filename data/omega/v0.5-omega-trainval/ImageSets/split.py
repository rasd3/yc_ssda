import numpy as np
import sys
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pcdet.datasets.omega.omega_dataset import trainval_split

ratio = float(sys.argv[1])
version = str(sys.argv[2])
num = int(sys.argv[3])

if 'omega' in version:
    train_scenes = trainval_split[version]['train_scenes']
    val_scenes = trainval_split[version]['val_scenes']
elif version == 'v1.0-trainval':
    train_scenes = splits.train
    val_scenes = splits.val
elif version == 'v1.0-test':
    train_scenes = splits.test
    val_scenes = []
elif version == 'v1.0-mini':
    train_scenes = splits.mini_train
    val_scenes = splits.mini_val
else:
    raise NotImplementedError

inds  = np.random.choice(len(train_scenes), int(len(train_scenes)*ratio), replace=False)
newlines = []
for i in inds:
    newlines.append(train_scenes[i])

with open("train_%.2f_%d.txt" % (ratio, num), "w") as fw:
    fw.write('\n'.join(newlines))

#  with open("train.txt", "r") as f:
    #  lines = f.read().strip().split('\n')
    #  inds = np.random.choice(len(lines), int(len(lines)*ratio), replace=False)
    #  newlines= []
    #  for i in inds:
        #  newlines.append(f'{lines[i]} {i}')
#
#  with open("train_%.2f_%d.txt" % (ratio, num), "w") as fw:
    #  fw.write('\n'.join(newlines))

