import sys; sys.path.append(".")


import os
from pathlib import Path
from pprint import pprint

import torch
import pytorch_lightning as pl
import numpy as np
import imageio
import moviepy.editor as mp

from scipy.spatial.transform import Rotation as RRR
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mGPT.render.pyrender.smpl_render import SMPLRender

# Notice: 使用 webui 保存的npy文件矩阵维度不正确（待查证）
PHASE = "demo"
TASK = "t2m"

# FEATS 用于生成 SMPL，JOINTS 用于生成 BHV
FEATS_NAME = "feats.npy"
JOINTS_NAME = "joints.npy"
GIF_NAME = "video.gif"
MP4_NAME = "video.mp4"

# Text2Model Bot
class T2MBot:
    def __init__(self, folder: str='cache'):
        cfg = parse_args(phase=PHASE) 
        cfg.FOLDER = folder

        # mkdir folder
        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(cfg.SEED_VALUE)

        # set the device to generate motion
        if cfg.ACCELERATOR == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        # create model object
        datamodule = build_data(cfg, phase="test")
        state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
        
        model = build_model(cfg, datamodule)
        model.load_state_dict(state_dict)
        model.to(device)

        # self
        self.cfg = cfg
        self.model = model
        self.output_dir = output_dir

        self.folder_name = ""

    def generate_motion(self, input: str, id: str, method: str="fast"):
        batch = {
            "text": [ input ],
            "length": [0]
        }

        outputs = self.model(batch, task=TASK)

        # wrap result
        feats = outputs["feats"][0]
        lengths = outputs["length"][0]
        joints = outputs["joints"][0]
        
        # 保存结果

        # 1. 创建文件，如果存在则删除该文件所有内容
        self.mkdir4result(id)

        print(joints.shape)

        # 2. 保存处理骨架信息 (joints)
        xyz = joints[:lengths]
        xyz = xyz[None]

        try:
            xyz = xyz.detach().cpu().numpy()
        except:
            xyz = xyz.detach().numpy()
            
        np.save(self.get_file_path(JOINTS_NAME), xyz)

        self.render_motion(
            joints,
            feats.to('cpu').numpy(), 
            method
        )
    
    def render_motion(self, data, feats, method='fast'):
        # 生成对应文件的路径
        npy_path, mp4_path, gif_path = \
            self.get_file_path(FEATS_NAME), self.get_file_path(MP4_NAME), self.get_file_path(GIF_NAME)

        # 保存npy文件
        np.save(npy_path, feats)

        if method == 'slow':
            if len(data.shape) == 4:
                data = data[0]
            data = data - data[0, 0]
            pose_generator = HybrIKJointsToRotmat()
            pose = pose_generator(data)
            pose = np.concatenate([
                pose,
                np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
            ], 1)
            shape = [768, 768]
            render = SMPLRender(self.cfg.RENDER.SMPL_MODEL_PATH)

            r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
            pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
            vid = []
            aroot = data[[0], 0]
            aroot[:, 1] = -aroot[:, 1]
            params = dict(pred_shape=np.zeros([1, 10]),
                        pred_root=aroot,
                        pred_pose=pose)
            render.init_renderer([shape[0], shape[1], 3], params)
            for i in range(data.shape[0]):
                renderImg = render.render(i)
                vid.append(renderImg)

            out = np.stack(vid, axis=0)
            imageio.mimwrite(gif_path, out, duration=50)
            out_video = mp.VideoFileClip(gif_path)
            out_video.write_videofile(mp4_path)
            del out, render

        elif method == 'fast':
            if len(data.shape) == 3:
                data = data[None]
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            pose_vis = plot_3d.draw_to_batch(data, [''], [gif_path])
            out_video = mp.VideoFileClip(gif_path)
            out_video.write_videofile(mp4_path)
            del pose_vis


    # 方便保存结果的封装函数
    def mkdir4result(self, id):
        try:
            folder_name = Path.joinpath(self.output_dir, id)
            folder_name.mkdir(parents=True)
        except FileExistsError:
            for filename in folder_name.iterdir():
                filename.unlink()
        finally:
            self.folder_name = folder_name

    def get_file_path(self, name):
        # Notice: 此处需要转换为字符串，否则会报错
        return str(Path.joinpath(self.folder_name, name))
