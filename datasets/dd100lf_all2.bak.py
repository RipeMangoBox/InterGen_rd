import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from utils.utils import process_dd100lf_motion_np, rigid_transform_dd100lf, qmul_np, qinv_np, qrot_np

joint_num = 22

# def custom_collate_fn(batch):
#     # batch is a list of tuples: [(name, music, pos3dl, pos3df, rot6dl, rot6df, gt_length), ...]
    
#     # Extract each component
#     names = [item[0] for item in batch]  # List of strings
#     music = [item[1] for item in batch]  # List of music tensors
#     lmotion = [item[2] for item in batch]  # List of pos3dl tensors
#     fmotion = [item[3] for item in batch]  # List of pos3df tensors
#     min_lens = [item[6] for item in batch]  # List of integers

#     # Stack tensors (assumes they have consistent shapes due to padding in __getitem__)
#     music = torch.stack(music, dim=0)  # Shape: [batch_size, seq_len, music_dim]
#     lmotion = torch.stack(lmotion, dim=0)  # Shape: [batch_size, seq_len, 75]
#     fmotion = torch.stack(fmotion, dim=0)  # Shape: [batch_size, seq_len, 75]
#     min_lens = torch.stack(min_lens, dim=0)  # Shape: [batch_size]

#     # Return a dictionary or tuple, depending on your needs
#     return {
#         'names': names,
#         'music': music,
#         'lmotion': lmotion,
#         'fmotion': fmotion,
#         'min_len': min_lens  # Keep as a list, or convert to tensor if needed: torch.tensor(min_lens)
#     }
    
def swap_left_right_position(data):
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    # left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 25, 56]
    # right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]
    left_hand_chain = [i for i in range(25, 40)]
    right_hand_chain = [i for i in range(40, 25)]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right(data):
    T, n_joints = data.shape[0], data.shape[1] // 3
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    positions = swap_left_right_position(positions).reshape(T, -1)

    return positions


class DD100lfAll2(Dataset):
    def __init__(self, music_root, motion_root, split='train', full_length=False):
        self.dances = {'pos3dl':[], 'pos3df':[], 'music':[]}
        dtypes = ['pos3d']
        self.names = []
        
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 240
        self.min_gt_length = 15
        self.full_length = full_length
        
        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1
        
        music_files = {}
        agent_files = {'leader':{}, 'follower':{}}

        def _process_files(agent_files, np_music):       
            this_pair = {}
            for agent in agent_files:
                # For each dtype_folder, load the corresponding file
                for dtype_folder in dtypes:
                    dance_path = agent_files[agent][take].replace('pos3d', dtype_folder)
                    if not os.path.isfile(dance_path):
                        continue
                    np_dance = np.load(dance_path)
                    np_dance = np_dance[:len(np_dance) - len(np_dance)%4, :] # to fit encodec down sample stragegy
                    this_pair[(agent, dtype_folder)] = np_dance
                
            for dtype_folder in dtypes:
                if (('leader', dtype_folder) not in this_pair) or (('follower', dtype_folder) not in this_pair):
                    continue
                ldance = this_pair[('leader', dtype_folder)]
                fdance = this_pair[('follower', dtype_folder)]
                min_len = min(len(ldance), len(fdance))
                ldance = ldance[:min_len, :joint_num*3]
                fdance = fdance[:min_len, :joint_num*3]
                
                seq_len = min(len(ldance), len(fdance))

                self.dances[dtype_folder+'l'].append(ldance[:seq_len])
                self.dances[dtype_folder+'f'].append(fdance[:seq_len])
                self.dances['music'].append(np_music[:seq_len//4*4])
                self.names.append(take)
        
        fnames = os.listdir(os.path.join(motion_root, 'pos3d', split))
        mnames = os.listdir(os.path.join(music_root,  'feature', split))
                
        for mname in mnames:
            path = os.path.join(music_root, 'feature', split, mname)
            music_files[mname[:-4]] = path
            
        for fname in fnames:
            path = os.path.join(motion_root, 'pos3d', split, fname)
            if path.endswith('_00.npy'):
                agent_files['follower'][fname[:-7]] = path
            elif path.endswith('_01.npy'):
                agent_files['leader'][fname[:-7]] = path
                
        for take in agent_files['follower']:
            if take not in agent_files['leader'] or take not in music_files:
                continue
            # music:
            music_path = music_files[take]
            np_music = np.load(music_path).astype(np.float32)
            # For each dtype, process files
            _process_files(agent_files, np_music)
        
        # 将长度对齐操作移到__init__阶段
        # 对self.dances中的所有序列进行长度对齐，裁剪到最小长度的最近的4的倍数
        # 合并为一次循环，使用 (k, v) 方式遍历
        keys = ['pos3dl', 'pos3df', 'music']
        for i in range(len(self.dances['pos3dl'])):
            min_len = min(len(self.dances[k][i]) for k in keys)
            # min_len = min_len - min_len % 4 + 1 # 提前+1，rigid_transform_dd100lf会自动drop掉最后一个
            min_len = min_len - min_len % 4
            for k in keys:
                self.dances[k][i] = self.dances[k][i][:min_len]
            
        print('DD100lfAll2 dataset loaded!')   

    def __len__(self):
        return len(self.dances['pos3dl'])
    
    def __getitem__(self, index):
        dances_motion = self.dances
        
        # 获取名称, 运动和音乐数据
        name = self.names[index]
        lmotion = dances_motion['pos3dl'][index]  # 领导者运动，形状 (seq_len, 75)
        fmotion = dances_motion['pos3df'][index]  # 跟随者运动，形状 (seq_len, 75)
        music = self.dances['music'][index]    # 音乐特征，形状 (seq_len, music_dim)
        
        length = lmotion.shape[0]
        if not self.full_length:
            if length > self.max_length:
                idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
                gt_length = self.max_gt_length
                lmotion = lmotion[idx:idx + gt_length]
                fmotion = fmotion[idx:idx + gt_length]
            else:
                raise ValueError(f"length {length} is less than max_length {self.max_length}")
                idx = 0
                gt_length = min(length - idx, self.max_gt_length )
                lmotion = lmotion[idx:idx + gt_length]
                fmotion = fmotion[idx:idx + gt_length]
        
        lmotion, root_quat_init1, root_pos_init1 = process_dd100lf_motion_np(lmotion, 0.001, 0, n_joints=22)
        fmotion, root_quat_init2, root_pos_init2 = process_dd100lf_motion_np(fmotion, 0.001, 0, n_joints=22)
        
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        # 与rigid_transform_dd100lf的配套一
        angle = np.arctan2(r_relative[:, 3:4], r_relative[:, 0:1])

        xy = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 1]]
        relative = np.concatenate([angle, xy], axis=-1)[0]
        fmotion = rigid_transform_dd100lf(relative, fmotion) # 丢弃最后一帧，长度变回self.max_gt_length
        
        gt_length = len(fmotion) # last frame dropped in process_dd100lf_motion_np, so we need to update the min_len
        music = music[:gt_length]
        
        # 零填充到 max_gt_length
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = lmotion.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            lmotion = np.concatenate((lmotion, padding_zeros), axis=0)
            fmotion = np.concatenate((fmotion, padding_zeros), axis=0)

            D_music = music.shape[1]
            padding_zeros_music = np.zeros((padding_len, D_music))
            music = np.concatenate((music, padding_zeros_music), axis=0)

        gt_length = len(fmotion)
        # Convert NumPy arrays to PyTorch tensors
        lmotion = torch.from_numpy(lmotion).float()
        fmotion = torch.from_numpy(fmotion).float()
        music = torch.from_numpy(music).float()
        gt_length = torch.tensor(gt_length).long()    
                        
        assert lmotion.shape[0] == fmotion.shape[0] == music.shape[0]
        
        return name, music, lmotion, fmotion, gt_length