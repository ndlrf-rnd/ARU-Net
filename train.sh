set -ex
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/vvasin/miniconda3/pkgs/cudatoolkit-10.0.130-0/lib"

export ROOT_DIR="/home/vvasin/ikutukov/ARU-Net"

export DATA_DIR="/home/vvasin/ikutukov/text-baselines"

for dataset in ${DATA_DIR}/train/*; do
    model="bd-$(basename ${dataset})-300dpi-256x10st"
    echo "Dataset: ${dataset}"
    echo "Model name: ${model}"
    
    if [[ ! -f "${ROOT_DIR}/models/${model}/model10.index" ]]; then
      python -u pix_lab/main/train_aru.py \
        --path_list_train "${dataset}/*.jpeg" \
        --path_list_val "${DATA_DIR}/train/**/*.jpeg" \
        --output_folder "${ROOT_DIR}/models/${model}/" \
        --restore_path "${ROOT_DIR}/models/${model}/" \
        --thread_num 1 \
        --queue_capacity 1 \
        --steps_per_epoch 256 \
        --epochs 10 \
        --max_val 256 \
        --scale_min 0.4 \
        --scale_max 1.0 \
        --scale_val 0.66 \
        --seed 13 \
        --gpu_device 0
    else
	    echo "Model already exists"
    fi

done




# python -u pix_lab/main/train_aru.py \
# 	--path_list_train '/home/vvasin/ikutukov/text-baselines/train/cbad-2017-*/*.jpeg' \
# 	--path_list_val '/home/vvasin/ikutukov/text-baselines/test/**/*.jpeg' \
# 	--output_folder '${ROOT_DIR}/models/aru-net-cbad-2017-2560x10/' \
# 	--restore_path '${ROOT_DIR}/models/aru-net-cbad-2017-2560x10/' \
# 	--thread_num 4 \
# 	--queue_capacity 128 \
# 	--steps_per_epoch 512 \
# 	--epochs 100 \
# 	--max_val 256 \
# 	--scale_min 0.4 \
# 	--scale_max 1.0 \
# 	--scale_val 0.66 \
# 	--seed 13
# # 	216 + 249

# python -u pix_lab/main/train_aru.py \
# 	--path_list_train '/home/vvasin/ikutukov/text-baselines/train/bozen-2016/*.jpeg' \
# 	--path_list_val '/home/vvasin/ikutukov/text-baselines/test/**/*.jpeg' \
# 	--output_folder '${ROOT_DIR}/models/aru-net-bozen-2016-2560x10/' \
# 	--restore_path '${ROOT_DIR}/models/aru-net-bozen-2016-2560x10/' \
# 	--thread_num 4 \
# 	--queue_capacity 128 \
# 	--steps_per_epoch 512 \
# 	--epochs 100 \
# 	--max_val 256 \
# 	--scale_min 0.4 \
# 	--scale_max 1.0 \
# 	--scale_val 0.66 \
# 	--seed 13
# # 350

# python -u pix_lab/main/train_aru.py \
# 	--path_list_train '/home/vvasin/ikutukov/text-baselines/train/cbad-2019/*.jpeg' \
# 	--path_list_val '/home/vvasin/ikutukov/text-baselines/test/**/*.jpeg' \
# 	--output_folder '${ROOT_DIR}/models/aru-net-cbad-2019-2560x10/' \
# 	--restore_path '${ROOT_DIR}/models/aru-net-cbad-2019-2560x10/' \
# 	--thread_num 4 \
# 	--queue_capacity 128 \
# 	--steps_per_epoch 512 \
# 	--epochs 100 \
# 	--max_val 256 \
# 	--scale_min 0.4 \
# 	--scale_max 1.0 \
# 	--scale_val 0.66 \
# 	--seed 13
# # 736
