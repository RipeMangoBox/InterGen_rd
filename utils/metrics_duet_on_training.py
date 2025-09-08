import numpy as np
import pickle 
from scipy import linalg
import os 
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import json
from collections import defaultdict
import pandas as pd
import re

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPLX_FEAT_POINT = [0, 7, 8, 10, 11, 15, 16, 17, 20, 21]


SMPLX_JOINT_NAMES = [
    'pelvis', #0
    'left_hip', 
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle', # 7
    'right_ankle', # 8
    'spine3', 
    'left_foot', # 10
    'right_foot', # 11
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow', 
    'right_elbow',
    'left_wrist', #20
    'right_wrist', #21
    'jaw', #22
    'left_eye_smplhf', #23
    'right_eye_smplhf', #24
    'left_index1', #25
    'left_index2', #26
    'left_index3', #27
    'left_middle1', #28
    'left_middle2', #29
    'left_middle3', #30
    'left_pinky1', #31
    'left_pinky2', #32
    'left_pinky3', #33 
    'left_ring1', #34
    'left_ring2',# 35
    'left_ring3', #36
    'left_thumb1', #37
    'left_thumb2', #38
    'left_thumb3', #39
    'right_index1', #40
    'right_index2', 
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

class DuetMetrics:
    def __init__(self):
        gt_pkl_root = '../ReactDance/data_lazy/motion/pos3d/test'
        gt_duet_freatures = [np.load(os.path.join(gt_pkl_root, 'duet_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'duet_features'))]
        self.stacked_gt_duet_features = np.stack(gt_duet_freatures)
        self.gt_jointsf_dict = {pkl: np.load(os.path.join(gt_pkl_root, pkl))[..., :66] for pkl in os.listdir(gt_pkl_root) if pkl.endswith('.npy')}

    def duet_feature(self, posef, posel):
        """
            posef: Tx55x3
            posel: Tx55x3
        """
        # Tx10x3
        Tf, _, _ = posef.shape
        Tl, _, _ = posel.shape
        T = np.min([Tf, Tl])
        posef = posef.copy()[:T, :]
        posel = posel.copy()[:T, :]
        feat = np.sqrt(np.sum((posef[:, SMPLX_FEAT_POINT][:, None, :, :] - posel[:, SMPLX_FEAT_POINT][:, :, None, :])**2, axis=-1)).reshape(T, -1) # 由于进行了reshape，融合了l和f被扩充的维度，因此两种扩充计算等价
        feat = np.mean(feat, axis=0)

        return feat

    def compute_mpjpe(self, pred, target, scale=1000.0):
        """
        Compute MPJPE (Mean Per Joint Position Error) in millimeters.
        pred: [T, J, 3] predicted 3D joint coordinates
        target: [T, J, 3] ground truth 3D joint coordinates
        scale: unit conversion factor (e.g., meters to millimeters)
        """
        assert pred.shape == target.shape
        error = np.sqrt(np.sum((pred - target)**2, axis=-1))  # [T, J]
        return np.mean(error) * scale
    
    def compute_mpjve(self, pred, target, scale=1000.0):
        """
        Compute standard MPJVE (Mean Per Joint Velocity Error) in millimeters (per frame).
        pred: [T, J, 3] predicted 3D joint coordinates (numpy array)
        target: [T, J, 3] ground truth 3D joint coordinates (numpy array)
        scale: unit conversion factor (e.g., meters to millimeters)
        """
        assert pred.shape == target.shape
        assert pred.ndim == 3 and pred.shape[-1] == 3
        # Compute displacement differences: [T-1, J, 3]
        pred_diff = pred[1:] - pred[:-1]
        target_diff = target[1:] - target[:-1]
        # L2 norm per joint per frame: [T-1, J]
        error = np.sqrt(np.sum((pred_diff - target_diff)**2, axis=-1))
        # Mean over all frames and joints
        return np.mean(error) * scale

    def compute_jitter(self, pred, target, scale=0.1):
        """
        Compute jitter (mean L2 distance of velocity differences) in millimeters.
        pred: [T, J, 3] predicted 3D joint coordinates
        target: [T, J, 3] ground truth 3D joint coordinates
        scale: scaling factor for jitter (default from reference code)
        """
        assert pred.shape == target.shape
        pred_vel = pred[1:] - pred[:-1]  # [T-1, J, 3]
        target_vel = target[1:] - target[:-1]  # [T-1, J, 3]
        error = np.sqrt(np.sum((pred_vel - target_vel)**2, axis=-1))  # [T-1, J]
        return np.mean(error) * scale

    def quantized_metrics(self, pred_features, jointsf_list, fname_list):
        pred_features = np.stack(pred_features)  # Nx72 p40
        gt_freatures = self.stacked_gt_duet_features # N' x 72 N' >> N
        gt_freatures, pred_features = normalize(gt_freatures, pred_features)
        fid = self.calc_fid(pred_features, gt_freatures)
        div = self.calculate_avg_distance(pred_features)
        
        pred_jointsf_list = []
        gt_jointsf_list = []
        # Load joint data for MPJPE and jitter
        for fname, pred_jointsf in zip(fname_list, jointsf_list):
            pred_jointsf = pred_jointsf.reshape(-1, 22, 3)
            pred_jointsf = pred_jointsf - pred_jointsf[:, 0, None]
            fname = fname.replace('_00.npy', '')    # unified process for training metrics and after_training metrics
            gt_jointsf = self.gt_jointsf_dict[f'{fname}_00.npy'].reshape(-1, 22, 3)[:pred_jointsf.shape[0], :]
            gt_jointsf = gt_jointsf - gt_jointsf[:, 0, None]
            pred_jointsf_list.append(pred_jointsf)
            gt_jointsf_list.append(gt_jointsf)

        # Compute MPJPE and jitter for follower (joint3df) and leader (joint3dl)
        mpjpe_f, mpjve_f, jitter_f = [], [], []
        for pred_f, gt_f in zip(pred_jointsf_list, gt_jointsf_list):
            mpjpe_f.append(self.compute_mpjpe(pred_f, gt_f))
            mpjve_f.append(self.compute_mpjve(pred_f, gt_f))
            jitter_f.append(self.compute_jitter(pred_f, gt_f))
        
        metrics = {
            'fid_k': abs(fid),
            'div': div,
            'mpjpe': np.mean(mpjpe_f),
            'mpjve': np.mean(mpjve_f),
            'jitter': np.mean(jitter_f),
        }
        
        return metrics

    def calc_fid(self, kps_gen, kps_gt):
        mu_gen = np.mean(kps_gen, axis=0)
        sigma_gen = np.cov(kps_gen, rowvar=False)

        mu_gt = np.mean(kps_gt, axis=0)
        sigma_gt = np.cov(kps_gt, rowvar=False)

        mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

        diff = mu1 - mu2
        eps = 1e-5
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            # print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                # raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


    def calc_diversity(self, feats):
        feat_array = np.array(feats)
        n, c = feat_array.shape
        diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
        return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

    def calculate_avg_distance(self, feature_list, mean=None, std=None):
        feature_list = np.stack(feature_list)
        n = feature_list.shape[0]
        # normalize the scale
        if (mean is not None) and (std is not None):
            feature_list = (feature_list - mean) / std
        dist = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist += np.linalg.norm(feature_list[i] - feature_list[j])
        dist /= (n * n - n) / 2
        return dist

    def calc_feats_duet(self, jointsl_list, jointsf_list):
        duet_feature_list = []

        for jointsl, jointsf in zip(jointsl_list, jointsf_list):
            n_joints = jointsf.shape[-1]//3
            joint3df = jointsf.reshape([-1, n_joints, 3])
            joint3dl = jointsl.reshape([-1, n_joints, 3])

            duet_feature_list.append(self.duet_feature(joint3df, joint3dl))
            
        return duet_feature_list

    def calc_db(self, keypoints, name=''):
        n_joints = keypoints.shape[-1]//3
        keypoints = np.array(keypoints).reshape(-1, n_joints, 3)
        kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
        kinetic_vel = G(kinetic_vel, 5)

        motion_beats = argrelextrema(kinetic_vel, np.less)
        return motion_beats, len(kinetic_vel)

    def BA(self, music_beats, motion_beats):
        if len(motion_beats[0]) == 0:
            return 0
        ba = 0
        for bb in music_beats[0]:
            ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
        return (ba / len(music_beats[0]))

    def calc_duet_be_score(self, jointsl_list, jointsf_list):
        ba_scores = []

        for jointsl, jointsf in zip(jointsl_list, jointsf_list):
            dance_beatsl, lengthl = self.calc_db(jointsl)     
            dance_beatsf, lengthf = self.calc_db(jointsf)
            ba_scores.append(self.BA(dance_beatsl, dance_beatsf))
            
        return np.mean(ba_scores)
    
    def eval_duet_metrics(self, jointsl_list, jointsf_list, fname_list):
        duet_metrics = defaultdict(float)
        duet_features = self.calc_feats_duet(jointsl_list, jointsf_list)
        be_score = self.calc_duet_be_score(jointsl_list, jointsf_list)
        quantized_scores = self.quantized_metrics(duet_features, jointsf_list, fname_list)
        
        duet_metrics["be_score"] = be_score
        duet_metrics.update(quantized_scores)
        
        return duet_metrics

def get_roots_from_dirs(pred_dirs):
    result = []
    for pred_dir in pred_dirs:
        # 遍历根目录及其子目录
        for dirpath, dirnames, _ in os.walk(pred_dir):
            # 检查当前文件夹是否包含 'npy' 或 'pos3d' 子文件夹
            if 'pos3d_npy' in dirnames:
                result.append(os.path.join(dirpath, 'pos3d_npy'))  # 记录父文件夹路径

    return result

def shorten_path(pred_root, index):
    """
    Shorten a model path by removing specific prefix and suffix, and reformatting epoch.
    
    Args:
        pred_root (str): Original model path.
        index (int): Index for prefixing the shortened path.
    
    Returns:
        str: Shortened path with format 'index. path_with_epoch'.
    """
    shortened_path = pred_root
    # Remove prefix and suffix
    prefix = '../ReactDance/results/generated/follower_gen/'
    suffix = '/pos3d_npy'
    if shortened_path.startswith(prefix):
        shortened_path = shortened_path[len(prefix):]
    if shortened_path.endswith(suffix):
        shortened_path = shortened_path[:-len(suffix)]
    # Extract and reformat epoch
    epoch_match = re.search(r'epoch=(\d+)', shortened_path)
    if epoch_match:
        epoch = epoch_match.group(1)
        shortened_path = re.sub(r'fid_best_epoch=\d+-Eval_FID=[0-9.]+.ckpt', f'e{epoch}', shortened_path)
    return f"{index}. {shortened_path}"

def sort_metrics_df(df):
    """
    Sort DataFrame by FID (low), Diversity (high), MPJPE (low), Jitter (low), BE Score (high).
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'FID', 'Diversity', 'MPJPE (mm)', 'Jitter', 'BE Score'.
    
    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(
        by=['FID', 'Diversity', 'MPJPE (mm)', 'Jitter', 'BE Score'],
        ascending=[True, False, True, True, False]
    )
    
def test_metrics_model(pred_root):
    # 2. 实例化DuetMetrics
    print("--- 1. 使用真实GT数据初始化DuetMetrics ---")
    duet_metric_calc = DuetMetrics()
    print("✅ DuetMetrics 初始化完成。")

    # 3. 加载预测数据和相应的GT数据
    print(f"\n--- 2. 从 '{os.path.basename(pred_root)}' 加载数据 ---")
    jointsl_list, jointsf_list, fname_list = [], [], []

    # 筛选出所有以 '_00.npy' 结尾的预测文件（代表跟随者）
    pred_files = sorted([f for f in os.listdir(pred_root) if f.endswith('_00.npy')])

    for follower_fname in pred_files:
        leader_fname = follower_fname.replace('_00.npy', '_01.npy')
        leader_path = os.path.join(pred_root, leader_fname)
        
        # 确认对应的领导者文件在GT目录中存在
        if os.path.exists(leader_path):
            # 加载预测的跟随者动作
            pred_follower_path = os.path.join(pred_root, follower_fname)
            pred_follower = np.load(pred_follower_path)[..., :66].reshape(-1, 66)
            jointsf_list.append(pred_follower)

            # 加载对应的真实领导者动作
            gt_leader = np.load(leader_path)[..., :66].reshape(-1, 66)[:pred_follower.shape[0], :]
            jointsl_list.append(gt_leader)
            
            assert pred_follower.shape == gt_leader.shape, f"预测跟随者动作和真实领导者动作的形状不一致: {pred_follower.shape} != {gt_leader.shape}"

            # 添加文件名，用于后续计算MPJPE/Jitter时查找对应的GT跟随者
            fname_list.append(follower_fname)
        else:
            print(f"警告：在GT目录中未找到对应的领导者文件 '{leader_fname}'。跳过 '{follower_fname}'。")

    if not jointsf_list:
        print("\n错误：未找到有效的动作对。请确保预测目录和GT目录中的文件名能够匹配。")
        exit()
        
    print(f"✅ 成功加载 {len(jointsf_list)} 个预测序列及其对应的领导者序列。")

    # 4. 运行评估
    print("\n--- 3. 运行评估 ---")
    duet_metrics = duet_metric_calc.eval_duet_metrics(jointsl_list, jointsf_list, fname_list)
    print("✅ 评估完成。")

    # 5. 打印结果
    print(f"\n--- 📊 评估结果 ---")
    print(f"模型: {pred_root}")
    print("-" * 35)
    for key, value in duet_metrics.items():
        print(f"{key.upper():<10}: {value:.4f}")
    print("-" * 35)
    
    return duet_metrics
    
if __name__ == '__main__':
    gt_root = '../ReactDance/data_lazy/motion/pos3d/test'
    music_root = '../ReactDance/data_lazy/music/feature/test'
    pred_dirs = [
        # 'results/generated/follower_gen/version_7/',
        '../ReactDance/results/generated/follower_gen/version_3/fid_best_epoch=130-Eval_FID=5.30.ckpt/samples',
    ]
    pred_roots = [
        gt_root,  # gt
        '../ReactDance/experiments/follower_gpt/eval/npy/pos3d/ep0500',
        '../ReactDance/experiments/rl/eval/npy/pos3d/ep0050',
    ]
    pred_roots += get_roots_from_dirs(pred_dirs)
    table_data = []
    for i, pred_root in enumerate(pred_roots):
        metrics = test_metrics_model(pred_root)
        if metrics:
            metrics['model_path'] = shorten_path(pred_root, i)
            table_data.append({
                'Model Path': metrics['model_path'],
                'FID': metrics['fid_k'],
                'Diversity': metrics['div'],
                'MPJPE (mm)': metrics['mpjpe'],
                'MPJVE (mm)': metrics['mpjve'],
                'Jitter': metrics['jitter'],
                'BE Score': metrics['be_score']
            })
    
    if table_data:
        df = pd.DataFrame(table_data)
        df = sort_metrics_df(df)
        pd.set_option('display.float_format', '{:.4f}'.format)
        df['MPJPE (mm)'] = df['MPJPE (mm)'].map('{:.2f}'.format)
        df['MPJVE (mm)'] = df['MPJVE (mm)'].map('{:.2f}'.format)
        print("\nEvaluation Results (Sorted by FID↓, Diversity↑, MPJPE↓, Jitter↓, BE Score↑):")
        print(df.to_string(index=False))
        