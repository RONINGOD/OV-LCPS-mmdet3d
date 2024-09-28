import os
import numpy as np
from nuscenes.nuscenes import NuScenes
import pickle
from tqdm import tqdm

in_dir = '/share/home/22351170/data/nuscenes_openseg_features_use/samples/LIDAR_TOP'
out_dir = '/share/home/22351170/data/nuscenes_openseg_multiview_features_transfer/samples/LIDAR_TOP'
version = 'v1.0-trainval'
data_root = '/share/home/22351170/data/nuscenes'
# nusc = NuScenes(version=version,dataroot=data_root,verbose=True)
# os.makedirs(out_dir,exist_ok=True)

token_idx = 0
splits = ['train','val']
for split in splits:
    with open(f'{data_root}/nus_pkl/nuscenes_infos_{split}.pkl', 'rb') as f:
            nusc_data = pickle.load(f)['data_list']
    for index,sample_data in tqdm(enumerate(nusc_data)):
        info = nusc_data[index]
        scene_token = sample_data['token']
        lidar_name = info['lidar_points']['lidar_path'].split('.')[0]
        clip_feature_path = os.path.join(in_dir,lidar_name,'.npz')
        try:
            points_features = np.load(clip_feature_path)
            clip_features, point_mask = points_features['point_feat'], points_features['point_mask']
        except Exception as e:
            print('wrong id:',index,lidar_name,scene_token)
            continue