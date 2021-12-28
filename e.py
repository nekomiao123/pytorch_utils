# install by "pip install wandb"
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print

# 第一步，登录。从wandb官网注册，获得login key，替换xxx部分
wandb.login(key='xxx')

# 如果不希望结果同步到网页，则使用这一行。
# os.environ['WANDB_MODE'] = 'dryrun'

# 调用这个method来初始化。如果是分布式训练，只需要master node做
# 这里传进来的参数是OmegaConf的一个config。Omniconf支持值里面有变量
# OmegaConf.to_container(cfg, resolve=True)把cfg转换成一个dict
# 同时resolve参数把里面的变量替换成对应值
def wandb_init(cfg: DictConfig):
    wandb.init(
        project='niubi',
        group=cfg.exp_group,
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    # 记得把最终的配置存下来，便于以后复现
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))

# main函数，"@hydra"部分是hydra库解析配置文件的方式。具体请参看其文档
# 这里config_path参数指我的配置文件（yaml）在'configs'这个路径
# config_name参数指默认的配置是该路径下的“defaults.yaml”这个文件
# 实际运行时可以用命令行参数override部分配置
@hydra.main(config_path='configs', config_name='defaults')
def main(cfg):
    # pretty用来是print出的文字带颜色. 来自于rich这个库
    pretty.install()
    # 把OmniConf的cfg转成yaml，print出来
    print(OmegaConf.to_yaml(cfg))
    # ...

    wandb_init(cfg)

    # During training, record a data point in this way
    # step=epoch records the x value of the curves, data records the y values
    # 'data' is a dict. Each key creates a figure with that as the title
    wandb.log(step=epoch, data={'loss': metric.mean})

    # do this after training
    wandb.finish()