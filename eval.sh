gpu=$1
model_name=$2
data_dir=/home/data

# 检查输入的实验模型名称是否为空
if [ ! "$model_name" ]; then
    echo "modelname is none"
    exit 1
fi

# 检查实验模型是否存在
if [ ! -d "$model_name" ]; then
    echo "$model_name do not exist"
    exit 1
fi

# 把训练时的模型文件拷贝出来（防止现在代码有变动）
cp $model_name/main.py ./
cp $model_name/prepro.py ./
cp $model_name/model.py ./

echo "use gpu: $gpu"

export CUDA_VISIBLE_DEVICES=$gpu; python main.py \
    --model_type=bert \
    --do_eval \
    --input_eval_file=$data_dir/test.tsv \
    --output_eval_file=$model_name/test_eval.txt \
    --data_dir=$data_dir \
    --model_name_or_path=$model_name \
    --task_name=cls \
    --output_dir=$model_name \
    --max_seq_length=20 \
    --per_gpu_eval_batch_size=512 \
    --ouput_path=$model_name \
    --fp16
