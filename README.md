# pytorch_utils

example 是一个通用的示例文件，里面是空的

classify_leaves 是一个kaggle中树叶分类比赛的示例文件

new_classify_leaves_example 是一个kaggle中树叶分类比赛的示例文件，使用了hydra wandb等工具，实验记录更加完整

代码结构和实用的pytorch用法

代码结构:

- train.py  训练文件
- predict.py  预测文件
- dataset.py  操作数据集文件(主要是继承dataset这个类)
- model.py  网络架构
- configs/defaults.yaml 配置文件
- utils.py  实用工具文件
- general_utils.py  通用工具文件(放一些通用的类或者函数之类的)
- requirement.txt   

可以修改defaults配置文件来修改超参数，也可以直接用传入的形式，像这样:
```
python train.py num_epochs=10
```

实用的pytorch用法

https://www.nekokiku.cn/2021/06/04/CV-tricks/

参考仓库：

1. [PyTorch-style](https://github.com/IgorSusmelj/pytorch-styleguide)
2. [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template#project-structure)
3. [Torch-base](https://github.com/ahangchen/torch_base)

