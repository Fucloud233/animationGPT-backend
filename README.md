# AnimationGPT

本项目后端在 [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) 的基础之上修改而成的。

## 使用配置

由于本项目同时支持中文与英文的输入，但是 LLM 只支持英文输入，因此本项目需要调用外部 API 将中文翻译成英文。
本项目目前使用的是有道智云的[文本翻译 API](https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html)，
如果需要正常使用，请访问其官网注册账号，并将 APP_KEY 等相关密钥配置在[`configs/translate.json`](./configs/translate.json)中。其配置格式如下。

```json
{
    "appKey": "xxx",
    "appSecret": "xxx"
}
```

## OOP 封装

为了方便代码的后续开发，本项目使用 OOP (Object-oriented Programming) 的思想对模型操作进行封装，结果为[`T2MBot`](./bot/T2MBot.py)对象。该对象在创建时，会经历以下初始化步骤。

1. 加载项目中的配置文件，并生成创建生成目录
1. 设置`torch`种子并选择运算设备
1. 依次加载`data_module`, `state_dict`构建模型对象

此后，用户就可以调用`generate_motion`通过文本生成模型动作。
