
import sys
import pickle as pkl
from nuscenes.nuscenes import NuScenes

split_txt = sys.argv[1]
split_name = sys.argv[2]

scene_list = []
new_datas = []
with open('%s' % split_txt, 'r') as f:
    lines = f.readlines()
    for line in lines:
        scene_list.append(int(line.strip().split('-')[1]))

nusc = NuScenes(version='v0.5-omega-trainval', dataroot='./', verbose=True)
datas = pkl.load(open('nuscenes_infos_10sweeps_train.pkl', 'rb'))

for data in datas:
    scene_num = int(nusc.get('scene', nusc.get('sample', data['token'])['scene_token'])['name'].split('-')[1])
    if scene_num not in scene_list:
        continue

    new_datas.append(data)

with open('nuscenes_infos_10sweeps_train_%s.pkl' % split_name, 'wb') as f:
    pkl.dump(new_datas, f)
