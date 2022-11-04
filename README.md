# 目录

<!-- TOC -->

- [目录](#目录)
- [BEIT描述](#BEIT描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的BEIT训练](#ImageNet-1k上的BEIT训练)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [BEIT描述](#目录)

BEIT的流程如图1所示，它的上侧是一个dVAE模型，下侧是一个类似BERT的Encoder。dVAE由Tokenizer和Decoder组成，其中Tokenizer的作用是将图像的每个Patch编码成一个视觉标志，Decoder
的作用将视觉标志恢复成输入图像，dVAE的这一部分是借鉴了DALL-E的思想。BEIT的下面部分是BERT，它的输入是含有被掩码的图像的所有patch，预测的是dVAE生成的视觉标志，这一部分是借鉴的MAE的思想，不同的是MAE预测的是归一化后的图像细节。

# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─imagenet
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
 ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/others/mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```text
└── BEIT
    ├── eval.py									# 推理文件
    ├── export.py								# 权重导出文件
    ├── README_CN.md							# 说明文件
    ├── requriments.txt							# 
    ├── scripts
    │   ├── run_distribute_train_ascend.sh		# 卡Ascend910训练脚本
    │   ├── run_eval_ascend.sh					# Ascend测试脚本
    │   └── run_standalone_train_ascend.sh		# 单卡Ascend910训练脚本
    ├── src
    │   ├── args.py								# 参数文件
    │   ├── configs					
    │   │   ├── parser.py	
    │   │   ├── pth2ckpt				
    │   │   │   └── pth2ckpt.ipynb				# pytorch预训练权重转mindspore
    │   │   └── beit_base_patch16_224.yaml	# BEIT的配置文件
    │   ├── data
    │   │   ├── augment							# 数据增强文件
    │   │   │   ├── auto_augment.py
    │   │   │   ├── __init__.py
    │   │   │   ├── mixup.py
    │   │   │   ├── random_erasing.py
    │   │   │   └── transforms.py
    │   │   ├── data_utils						# obs交互文件
    │   │   │   ├── __init__.py
    │   │   │   └── moxing_adapter.py
    │   │   ├── imagenet.py						# ImageNet数据类
    │   │   └── __init__.py
    │   ├── models								# 模型定义
    │   │   ├── __init__.py
    │   │   ├── layers
    │   │   │   ├── drop_path.py
    │   │   │   └── identity.py
    │   │   └── beit.py
    │   ├── tools
    │   │   ├── callback.py						# 回调函数
    │   │   ├── cell.py							# 关于cell的自定义类
    │   │   ├── criterion.py					# 损失函数
    │   │   ├── get_misc.py						# 功能函数
    │   │   ├── __init__.py		
    │   │   ├── optimizer.py					# 优化器文件
    │   │   └── schedulers.py					# 学习率策略
    │   └── trainer
    │       └── train_one_step.py				# 自定义单步训练
    └── train.py								# 训练文件						

```

## 脚本参数

在beit_base_patch16_224.yaml中可以同时配置训练参数和评估参数。

- 配置BEIT和ImageNet-1k数据集。

  ```text
  # Architecture 85.2%
  arch: beit_base_patch16_224
  
  # ===== Dataset ===== #
  data_url: ../data/imagenet
  set: ImageNet
  num_classes: 1000
  mix_up: 0.
  cutmix: 0.
  auto_augment: rand-m9-mstd0.5-inc1
  interpolation: bicubic
  re_prob: 0.25
  re_mode: pixel
  re_count: 1
  mixup_prob: 1.0
  switch_prob: 0.5
  mixup_mode: batch
  image_size: 224
  crop_pct: 0.875
  
  
  # ===== Learning Rate Policy ======== #
  optimizer: adamw
  base_lr: 0.00002
  warmup_lr: 0.000001
  min_lr: 0.000001
  lr_scheduler: cosine_lr
  warmup_length: 5
  layer_decay: 0.85
  
  
  # ===== Network training config ===== #
  amp_level: O1
  keep_bn_fp32: True
  beta: [ 0.9, 0.999 ]
  clip_global_norm_value: 5.
  is_dynamic_loss_scale: True
  epochs: 30
  cooldown_epochs: 0
  label_smoothing: 0.1
  weight_decay: 0.00000001
  momentum: 0.9
  batch_size: 64
  drop_path_rate: 0.1
  pretrained: s3://open-data/beit/src/beit_base_patch16_224_pt22k_ft22k.ckpt
  
  # ===== Hardware setup ===== #
  num_parallel_workers: 16
  device_target: Ascend
  
  # ===== Model Config ===== #
  rel_pos_bias: True
  abs_pos_emb: False
  layer_scale_init_value: 0.1
  ```

更多配置细节请参考脚本`beit_base_patch16_224.yaml`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/beit_base_patch16_224.yaml \
  > train.log 2>&1 &
  
  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]
  
  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  
  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/beit_base_patch16_224.yaml \
  --pretrained ./ckpt_0/beit_base_patch16_224.ckpt > ./eval.log 2>&1 &
  
  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

  [hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的BEIT训练

| 参数                 | Ascend                          |
| -------------------------- |---------------------------------|
|模型| BEIT                            |
| 模型版本              | beit_base_patch16_224           |
| 资源                   | Ascend 910 8卡                   |
| 上传日期              | 2022-11-04                      |
| MindSpore版本          | 1.5.1                           |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像 |
| 训练参数        | epoch=30, batch_size=512        |
| 优化器                  | AdamWeightDecay                 |
| 损失函数              | SoftTargetCrossEntropy          |
| 损失| 1.502                           |
| 输出                    | 概率                              |
| 分类准确率             | 八卡：top1:85.27% top5:97.67%      |
| 速度                      | 8卡：546.446毫秒/步                  |
| 训练耗时          | 13h45min03s（run on OpenI）       |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)