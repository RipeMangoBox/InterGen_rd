import copy
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegFileWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import shutil
import subprocess
import tempfile

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
up_axis = 2

def plot_3d_motion_alternative(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4, up_axis=1):
    """
    通过先生成图像序列，再用 ffmpeg 合成视频的方式来可视化3D动作，以避免并发冲突。
    """
    # 1. 创建一个唯一的临时目录
    # tempfile.mkdtemp() 会创建一个唯一的临时文件夹，从根本上解决了并发冲突问题
    temp_dir = tempfile.mkdtemp()
    print(f"临时帧文件将保存在: {temp_dir}")

    # --- 您原有的数据预处理和设置代码 (基本不变) ---
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)
        ax.view_init(azim=-60, elev=30) # 固定视角
        ax.dist = 20
    
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig) # Axes3D is deprecated, use projection='3d'
    # ax = fig.add_subplot(111, projection='3d')
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])

    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_colors = [colors[i] for i in range(len(mp_joints))]

    for i, joints in enumerate(mp_joints):
        data = joints.copy().reshape(len(joints), -1, 3)
        height_offset = data.min(axis=0).min(axis=0)[up_axis]
        data[:, :, up_axis] -= height_offset
        mp_data.append({"joints": data})

    def plot_xyPlane(minx, maxx, miny, maxy, minz):
        ## Plot a plane XY
        verts = [
            [minx, miny, minz],
            [minx, maxy, minz],
            [maxx, maxy, minz],
            [maxx, miny, minz]
        ]
        xy_plane = Poly3DCollection([verts])
        xy_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xy_plane)
        
    # --- 核心修改部分 ---
    try:
        # 2. 循环生成每一帧图像
        for i in range(frame_number):
            # 清除上一帧的绘图内容
            ax.lines = []
            ax.collections = []
            plot_xyPlane(-4, 4, -4, 4, 0)
            # 绘制地面
            verts = [[-radius, -radius, 0], [-radius, radius, 0], [radius, radius, 0], [radius, -radius, 0]]
            ax.add_collection3d(Poly3DCollection([verts], facecolors='lightgrey', linewidths=0, alpha=0.2))

            # 绘制当前帧的骨架
            for pid, data in enumerate(mp_data):
                for chain, color in zip(kinematic_tree, [mp_colors[pid]] * len(kinematic_tree)):
                    linewidth = 2.0 if np.all(np.array(chain) < 5) else 1.0 # 简化了线宽逻辑
                    ax.plot3D(data["joints"][i, chain, 0],
                              data["joints"][i, chain, 1], # Y 轴
                              data["joints"][i, chain, 2], # Z 轴 (up)
                              linewidth=linewidth, color=color)

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # 将当前帧保存为PNG文件，文件名按顺序编号
            frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
            plt.savefig(frame_path, dpi=100) # dpi可以按需调整

            # 打印进度
            if (i+1) % 100 == 0:
                print(f"已生成 {i+1}/{frame_number} 帧...")

        # 3. 使用 ffmpeg 将图像序列合成为视频
        print("所有帧已生成, 开始使用 ffmpeg 合成视频...")
        # 确保输出目录存在
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        ffmpeg_cmd = [
            'ffmpeg',
            '-loglevel', 'verbose',
            '-r', str(fps),                   # 设置帧率
            '-i', f'{temp_dir}/frame_%05d.png', # 输入文件模式
            '-vcodec', 'libx264',             # 使用 H.264 编码器
            '-pix_fmt', 'yuv420p',            # 像素格式，确保在大多数播放器上兼容
            '-y',                             # 覆盖已存在的输出文件
            save_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"成功保存动画到 {save_path}")

    except Exception as e:
        print(f"生成动画时发生错误: {e}")
    finally:
        # 4. 清理临时文件夹
        print(f"清理临时文件夹: {temp_dir}")
        shutil.rmtree(temp_dir)

    plt.close(fig)


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)
        
    def plot_xyPlane(minx, maxx, miny, maxy, minz):
        ## Plot a plane XY
        verts = [
            [minx, miny, minz],
            [minx, maxy, minz],
            [maxx, maxy, minz],
            [maxx, miny, minz]
        ]
        xy_plane = Poly3DCollection([verts])
        xy_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xy_plane)
        
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i,joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        height_offset = MINS[up_axis]
        data[:, :, up_axis] -= height_offset
        trajec = data[:, 0, [0, 1]]

        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })

    def update(index):
        ax.lines = []
        ax.collections = []
        # ax.view_init(elev=120, azim=-90)
        ax.dist = 30#7.5
        #         ax =
        plot_xyPlane(-6, 6, -6, 6, 0)
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    writer = FFMpegFileWriter(fps=fps)
    # writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved animation to {save_path}")

def plot_t2m(mp_data, save_path, caption): # Changed result_path to save_path for clarity
    mp_joint = []
    for i, data in enumerate(mp_data):
        if i == 0:
            joint = data[:,:22*3].reshape(-1,22,3)
        else:
            joint = data[:,:22*3].reshape(-1,22,3)

        mp_joint.append(joint)

    # Pass the full path with the extension directly
    plot_3d_motion_alternative(save_path, t2m_kinematic_chain, mp_joint, title=caption, fps=30)

def generate_one_sample(motion_output_multiple, name, result_path):
    # Append the .mp4 extension to the result path
    # This is the fix!
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, f"{name}.mp4")
    
    # Pass the corrected path to the plotting function
    if motion_output_multiple[0].ndim == 2:
        plot_t2m(motion_output_multiple, save_path, name)
    elif motion_output_multiple[0].ndim == 3:
        plot_t2m([motion[0] for motion in motion_output_multiple], save_path, name)
    else:
        raise ValueError(f"Invalid number of motions: {len(motion_output_multiple)}")