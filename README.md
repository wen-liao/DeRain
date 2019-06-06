# DeRain

## 简介
代码复现了论文中的除雨算法。[[论文链接]](https://arxiv.org/abs/1711.10098)

## 参数设置
参数位于parameters.py文件中
下载数据集 https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K

DISCRIMINATOR：使用判别器
ATTENTIVE_DISCRIMINATOR：使用注意力判别器
ATTENTIVE_AUTOENCODER：使用带有注意力机制的自编码器

TRAIN_PATH, EVAL_PATH, TEST_PATH：数据集目录
BATCH_SZ：每个batch中的图片数目
THETA, GAMMA, LAMBDA：论文中模型的超参数
LR：学习率
ITERATION：LSTM中单元个数
EPOCHES：训练时的epoch个数

MODEL_PATH：测试模型性能时模型的位置

device：训练时的单元，可以设置为'cpu','cuda:0'等

#训练模型
在parameters.py中设置好参数。运行命令 python train.py。在images文件夹中保存的图片显示了训练的效果。文件夹weights中保存了模型的参数。

#测试模型性能
在parameters.py中设置好参数。运行命令python evaluate.py可以测试模型的性能。
