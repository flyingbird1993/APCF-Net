# 模型与配置说明

本工程实现《烟草病害细粒度数据集构建与注意力图块互补识别方法》中的APCF-Net。正式数据集为FTDD，共17类，图像和CSV由作者通过README中的百度网盘提供。

## 模块对应

| 论文组成 | 实现文件 | 主要行为 |
|---|---|---|
| ViT-B/16主干 | apcf/backbone.py | 16×16图块、768维、12层Transformer |
| AAPS | apcf/aaps.py | 全层残差注意力累积、保留高注意力图块、批内互补组合 |
| MCFF | apcf/model.py、apcf/mcff.py | 第10/11/12层独立分类、等权监督、概率融合 |
| CAM | apcf/cam.py | MSA后的图块令牌重塑，水平/垂直方向门控 |
| 训练与评估 | train.py、evaluate.py、apcf/engine.py | 训练集更新、验证集选模、独立测试 |
| 配置与数据 | configs/paper.json、apcf/data.py | 17类、224输入、正式CSV读取 |

主干与模块来自提供的 `CA_CTI_APC.py` 和 `coordatt.py`，训练流程参考 `main_CA_CTI_APC.py`。原始文件指纹保留在 `metadata/source_inventory.json`。

## 训练设置

论文明确给出的设置：Adam、初始学习率3e-4、60轮、batch64、阶梯衰减、AAPS训练触发概率0.5、MCFF分支等权1/3。推理阶段无图块交换。

配置同时显式列出实现细节：step_size=20、gamma=0.1、weight_decay=0、裁剪前resize=256、CAM reduction=32、AAPS grid=4、mixed_loss_weight=1。其中论文未具体列出的数值为整理工程的默认选择，可按正式实验设置调整；并不据此推断作者当时使用的每项参数。

AAPS标签权重按论文式(6)与提供代码计算保留图块的注意力总和，默认不二次归一化。`normalize_mix_weights=true` 为可选实验变化。

## 数据路径

正式配置约定使用 `data/tobacco/images` 和 `data/tobacco/{train,val,test}.csv`。下载文件也可保留原位置，修改配置指向真实路径。类别编号与名称以正式CSV为准，不由工程重新推断或合并。

数据包含三级标注；当前三个预测头均预测同一17类病害任务，并非三个不同标签层级。

## 工程支持

- CPU/GPU随输入设备运行，不使用硬编码GPU编号。
- 模型统一返回具名输出；NPZ加载路径使用跨平台键名。
- 训练和验证清单检查标签、文件、重复与交集；评估不丢弃尾批。
- 保存模型、优化器、学习率计划、随机状态和清单指纹，支持epoch边界恢复。
- 使用Python文件对象保存检查点，支持中文路径。

论文性能参见README实验结果；整理工程的本地运行检查见 [VALIDATION.md](VALIDATION.md)。
