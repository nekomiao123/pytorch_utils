# 使用方法 bash train.sh gpu_id modelname
gpu=$1
model_name=$2
pretrained_model=/home/pretrained_model
data_dir=/home/data

# 检查输入的实验模型名称是否为空
if [ ! "$model_name" ]; then
    echo "modelname is none"
    exit 1
fi

# 创建实验文件夹
if [ ! -d "$model_name" ]; then
    mkdir $model_name
fi

# 检查实验是否已存在
if [ -n "`find $model_name -maxdepth 1 -name '*.bin'`" ]; then
    echo "model exists"
    exit 1
fi

# 备份实验代码
cp main.py $model_name/
cp prepro.py $model_name/
cp model.py $model_name/
cp train.sh $model_name/

echo "use gpu: $gpu"

export CUDA_VISIBLE_DEVICES=$gpu; nohup python -u main.py \
    --model_type=bert \
    --data_dir=$data_dir \
    --input_train_file=train.tsv \
    --input_eval_file=$data_dir/dev.txt\
    --output_eval_file=$model_name/dev_eval.txt  \
    --model_name_or_path=$pretrained_model \
    --task_name=cls \
    --output_dir=$model_name \
    --max_seq_length=20 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size=512 \
    --per_gpu_eval_batch_size=512 \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --warmup_steps=10000 \
    --num_train_epochs=2 \
    --logging_steps=5000 \
    --save_steps=5000 \
    --ouput_path=$model_name \
    --do_lower_case \
    --fp16 \
    > $model_name/log.txt 2>&1 & # 保存log

echo "$model_name/log.txt"