config=$1
model_path=$2
output=$3
dataset_path=$4

python demo/merge_id_uet.py --config-file $config \
	--parallel --vis-label --dataset-name VTXReidGroup --output $output \
	--dataset-path $dataset_path \
	--opts MODEL.WEIGHTS $model_path
