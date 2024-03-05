import time
import os
from pathlib import Path

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

PHASE = "webui"
TASK = "t2m"

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

    def generate_motion(self, input: str, method: str="fast"):
        # TODO: unknown parameters
        motion_length = 0
        motion_token_string = ""

        prompt = self.model.lm.placeholder_fulfill(
            input, 
            motion_length,
            motion_token_string, 
            ""
        )

        result = {
            'model_input': prompt
        }

        batch = {
            "length": [ motion_length ],
            "text": [ prompt ]
        }

        outputs = self.model(batch, task=TASK)

        # wrap result
        out_feats = outputs["feats"][0]
        out_lengths = outputs["length"][0]
        out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
        out_texts = outputs["texts"][0]
        output_mp4_path, video_fname, output_npy_path, joints_fname = \
        self.render_motion(
            out_joints,
            out_feats.to('cpu').numpy(), method
        )
        
        result['model_output'] = {
            "feats": out_feats,
            "joints": out_joints,
            "length": out_lengths,
            "texts": out_texts,
            "motion_video": output_mp4_path,
            "motion_video_fname": video_fname,
            "motion_joints": output_npy_path,
            "motion_joints_fname": joints_fname,
        }

        return result
    
    
    def render_motion(self, data, feats, method='fast'):
        fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
            time.time())) + str(np.random.randint(10000, 99999))
        video_fname = fname + '.mp4'
        feats_fname = fname + '.npy'
        output_npy_path = os.path.join(self.output_dir, feats_fname)
        output_mp4_path = os.path.join(self.output_dir, video_fname)
        np.save(output_npy_path, feats)

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
            output_gif_path = output_mp4_path[:-4] + '.gif'
            imageio.mimwrite(output_gif_path, out, duration=50)
            out_video = mp.VideoFileClip(output_gif_path)
            out_video.write_videofile(output_mp4_path)
            del out, render

        elif method == 'fast':
            output_gif_path = output_mp4_path[:-4] + '.gif'
            if len(data.shape) == 3:
                data = data[None]
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
            out_video = mp.VideoFileClip(output_gif_path)
            out_video.write_videofile(output_mp4_path)
            del pose_vis

        return output_mp4_path, video_fname, output_npy_path, feats_fname


