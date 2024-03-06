import sys; sys.path.append(".")

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

MP4_NAME = "video.mp4"
FEATS_NAME = "feats.npy"
GIF_NAME = "video.gif"

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

    def generate_motion(self, input: str, id: str, method: str="fast"):
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
        self.render_motion(
            out_joints,
            out_feats.to('cpu').numpy(), 
            id,
            method
        )
        
        result['model_output'] = {
            "feats": out_feats,
            "joints": out_joints,
            "length": out_lengths,
            "texts": out_texts,
            # "motion_video": output_mp4_path,
            # "motion_video_fname": video_fname,
            # "motion_joints": output_npy_path,
            # "motion_joints_fname": joints_fname,
        }

        return result
    
    def render_motion(self, data, feats, fname: str, method='fast'):
        # 根据时间自动生成文件名
        # if fname is None:
        #     fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        #         time.time())) + str(np.random.randint(10000, 99999))
        
        # 创建文件，如果存在则删除该文件所有内容
        try:
            folder_name = Path.joinpath(self.output_dir, fname)
            folder_name.mkdir(parents=True)
        except FileExistsError:
            for filename in folder_name.iterdir():
                filename.unlink()
        
        # 生成对应文件的路径
        # Notice: 此处需要转换为字符串，否则会报错
        npy_path = str(Path.joinpath(folder_name, FEATS_NAME))
        mp4_path = str(Path.joinpath(folder_name, MP4_NAME))
        gif_path = str(Path.joinpath(folder_name, GIF_NAME))

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