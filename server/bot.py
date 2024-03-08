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

# Notice: 使用 webui 保存的npy文件举证维度不正确（待查证）
PHASE = "demo"
TASK = "t2m"

MP4_NAME = "video.mp4"
FEATS_NAME = "feats.npy"
GIF_NAME = "video.gif"


# 来自 demo.py 的代码，用于处理输入文本（意义不明）
def load_example_input(lines, model):
    def motion_token_to_string(motion_token, lengths, codebook_size=512):
        motion_string = []
        for i in range(motion_token.shape[0]):
            motion_i = motion_token[i].cpu(
            ) if motion_token.device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{codebook_size}>' +
                ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                f'<motion_id_{codebook_size + 1}>'))
        return motion_string

    lines = [line for line in lines if line.strip()]
    count = 0
    texts = []
    
    # Strips the newline character
    motion_joints = [torch.zeros((1, 1, 22, 3))] * len(lines)
    motion_lengths = [0] * len(lines)
    motion_token_string = ['']

    motion_head = []
    motion_heading = []
    motion_tailing = []

    motion_token = torch.zeros((1, 263))
    
    for i, line in enumerate(lines):
        count += 1
        if len(line.split('#')) == 1:
            texts.append(line)
        # Notice 如果输入文本中不包含符号 '#'，则以下的代码没有任何意义
        else:
            feat_path = line.split('#')[1].replace('\n', '')
            if os.path.exists(feat_path):
                feats = torch.tensor(np.load(feat_path), device=model.device)
                feats = model.datamodule.normalize(feats)

                motion_lengths[i] = feats.shape[0]
                motion_token, _ = model.vae.encode(feats[None])

                motion_token_string = motion_token_to_string(
                    motion_token, [motion_token.shape[1]])[0]
                motion_token_length = motion_token.shape[1]

                motion_splited = motion_token_string.split('>')

                split = motion_token_length // 5 + 1
                split2 = motion_token_length // 4 + 1
                split3 = motion_token_length // 4 * 3 + 1

                motion_head.append(motion_token[:, :motion_token.shape[1] //
                                                5][0])

                motion_heading.append(feats[:feats.shape[0] // 4])

                motion_tailing.append(feats[feats.shape[0] // 4 * 3:])

                if '<Motion_Placeholder_s1>' in line:
                    motion_joints[i] = model.feats2joints(
                        feats)[:, :feats.shape[1] // 5]
                else:
                    motion_joints[i] = model.feats2joints(feats)

                motion_split1 = '>'.join(
                    motion_splited[:split]
                ) + f'><motion_id_{model.codebook_size+1}>'
                motion_split2 = f'<motion_id_{model.codebook_size}>' + '>'.join(
                    motion_splited[split:])

                motion_masked = '>'.join(
                    motion_splited[:split2]
                ) + '>' + f'<motion_id_{model.codebook_size+2}>' * (
                    split3 - split2) + '>'.join(motion_splited[split3:])

            texts.append(
                line.split('#')[0].replace(
                    '<motion>', motion_token_string).replace(
                        '<Motion_Placeholder_s1>', motion_split1).replace(
                            '<Motion_Placeholder_s2>', motion_split2).replace(
                                '<Motion_Placeholder_Masked>', motion_masked))

    return_dict = {
        'text': texts,
        'motion_joints': motion_joints,
        'motion_lengths': motion_lengths,
        'motion_token': motion_token,
        'motion_token_string': motion_token_string,
    }
    if len(motion_head) > 0:
        return_dict['motion_head'] = motion_head

    if len(motion_heading) > 0:
        return_dict['motion_heading'] = motion_heading

    if len(motion_tailing) > 0:
        return_dict['motion_tailing'] = motion_tailing

    return return_dict

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
        
        model = build_model(cfg, datamodule)
        model.to(device)

        # self
        self.cfg = cfg
        self.model = model
        self.output_dir = output_dir

    def generate_motion(self, input: str, id: str, method: str="fast"):
        # TODO: unknown parameters
        motion_length = 0
        motion_token_string = ""

        return_dict = load_example_input([input], self.model)
        text, in_joints = return_dict['text'], return_dict['motion_joints']

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
            "length": return_dict["motion_lengths"],
            "text": text
        }

        outputs = self.model(batch, task=TASK)

        # wrap result
        feats = outputs["feats"][0]
        lengths = outputs["length"]
        joints = outputs["joints"]
        texts = outputs["texts"]

        xyz = joints[0][:lengths[0]]
        xyz = xyz[None]

        try:
            xyz = xyz.detach().cpu().numpy()
            xyz_in = in_joints[0][None].detach().cpu().numpy()
        except:
            xyz = xyz.detach().numpy()
            xyz_in = in_joints[0][None].detach().numpy()

        # id = b * batch_size + i
            
        np.save("./cache/temp/out.npy", xyz)
        np.save("./cache/temp/in.npy", xyz_in)
    


        self.render_motion(
            joints,
            feats.to('cpu').numpy(), 
            id,
            method
        )
    
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

        print("feats: ", feats.shape)

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