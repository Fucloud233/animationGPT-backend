# AnimationGPT

本项目后端在 [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) 的基础之上修改而成的。

## OOP 封装

为了方便代码的后续开发，本项目使用 OOP (Object-oriented Programming) 的思想对模型操作进行封装，结果为[`T2MBot`](./bot/T2MBot.py)对象。该对象在创建时，会经历以下初始化步骤。

1. 加载项目中的配置文件，并生成创建生成目录
1. 设置`torch`种子并选择运算设备
1. 依次加载`data_module`, `state_dict`构建模型对象

此后，用户就可以调用`generate_motion`通过文本生成模型动作。
