# This is the example for classfy leaves

[The kaggle competition](https://www.kaggle.com/c/classify-leaves)

dataset is the file leaves_data

single-GPU 
python train.py

muti-GPU 
CUDA_VISIBLE_DEVICES=4,5 accelerate launch train.py

Architecture:

- train.py 
- predict.py 
- dataset.py 
- model.py 
- utils.py 
- general_utils.py 
