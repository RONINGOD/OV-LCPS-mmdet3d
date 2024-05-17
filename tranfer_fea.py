import os
import numpy as np
from nuscenes.nuscenes import NuScenes
import pickle
from tqdm import tqdm

in_dir = '/share/home/22351170/data/nuscenes_openseg_multiview_features/samples/LIDAR_TOP'
out_dir = '/share/home/22351170/data/nuscenes_openseg_multiview_features_transfer/samples/LIDAR_TOP'
version = 'v1.0-trainval'
data_root = '/share/home/22351170/data/nuscenes'
nusc = NuScenes(version=version,dataroot=data_root,verbose=True)
os.makedirs(out_dir,exist_ok=True)

token_idx = 0
splits = ['train','val']
for split in splits:
    with open(f'{data_root}/nus_pkl/nuscenes_infos_{split}.pkl', 'rb') as f:
            nusc_data = pickle.load(f)['data_list']
    for index,sample_data in tqdm(enumerate(nusc_data)):
        info = nusc_data[index]
        scene_token = sample_data['token']
        scene_rec = nusc.get('scene', scene_token)
        first_sample_token = scene_rec['first_sample_token']
        file_name = os.path.join(in_dir,first_sample_token,'.npz')
        print(file_name)
        print(first_sample_token)
        if os.path.exists(os.path.join(in_dir,first_sample_token,'.npz')):
            token_idx+=1

    if token_idx==len(nusc_data):
        print('{} is done'.format(split))