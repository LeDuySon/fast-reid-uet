config=$1
model_path=$2
output=$3

python demo/visualize_result.py --config-file $config \
	--parallel --vis-label --dataset-name VTXReid --output $output \
	--opts MODEL.WEIGHTS $model_path
