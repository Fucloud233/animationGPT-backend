# AnimationGPT

本项目后端在 [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) 的基础之上修改而成的。

## 使用配置

### 基础依赖

除了从 Github 克隆的代码之外，若需要正常使用本模型，
还需要下载和安装以下依赖文件。

1. 通过 `pip install -r requirement` 安装依赖
2. 下载项目所依赖的相关模型
3. 下载项目所需要的数据集 `Humanml3d`
    > 需要注意的是原项目是使用 [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 的数据集，
    > 但是其只提供了数据集`KIT_ML`，而该数据需要手动处理

原项目也提供了下载相关依赖的脚本，当然也支持从相关网站上手动下载

```bash
bash prepare/download_smpl_model.sh
bash prepare/prepare_t5.sh
bash prepare/download_t2m_evaluators.sh

bash prepare/download_pretrained_models.sh
```

-   依赖文件：[Google Drive](https://drive.google.com/drive/folders/10s5HXSFqd6UTOkW2OMNc27KGmMLkVc2L)
-   模型文件: [Huggingface](https://huggingface.co/OpenMotionLab)

### 翻译 API

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

为了方便代码的后续开发，本项目使用 OOP (Object-oriented Programming) 的思想对模型操作进行封装，结果为[`T2MBot`](./server/bot.py)对象。该对象在创建时，会经历以下初始化步骤。

1. 加载项目中的配置文件，并生成创建生成目录
1. 设置`torch`种子并选择运算设备
1. 依次加载`data_module`, `state_dict`构建模型对象

此后，用户就可以调用`generate_motion`通过文本生成模型动作。

## 缓存机制

为了防止用户对相同的句子重复生成，特别是对于我们提供了一些实例，
我们引入了缓存的机制，当用户请求生成的句子已经生成过，
服务端则直接返回结果，以此减少服务端的计算开销。

考虑到具体的业务量，服务端的缓存机制如下。

1. 服务端启动时，从缓存目录读取之前已经生成的结果 id 作为集合
2. 当有新的生成请求时，首先将 prompt 通过哈希算法得到对应的 id
3. 然后判断 id 在集合中是否存在
    - 存在则直接返回生成结果
    - 不存在则进行生成，并将该 id 保存到集合当中
4. 当集合中结果的数量超过设置的最大限制，则会进行缓存淘汰
    > 程序会随机删除 n% 最大数量的生成结果，兼顾效率和数据一致性的问题，
    > 程序会先删除内存中的集合，再通过多线程删除 磁盘中的生成结果

通过观察可以服务端生成的文件大小总和再 500K 左右，
如果将最大生成数量设置为 2000，会占用磁盘 1G 的空间大小。
