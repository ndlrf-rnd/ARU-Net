set -e

export STEPS="${STEPS:-256}"
export EPOCHS="${EPOCHS:-100}"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/vvasin/miniconda3/pkgs/cudatoolkit-10.0.130-0/lib"

export MODEL_NAME='aru'
export COST_NAME='cross_entropy'
export OPTIMIZER_NAME='rmsprop'

export SEED=${SEED:-13}
export TASK_NAME=${TASK_NAME:-'bd'}

MODEL_DIR_NAME="${MODEL_NAME}-${COST_NAME}-${OPTIMIZER_NAME}-${TASK_NAME}-prod"
MODEL_PATH="./models/${MODEL_DIR_NAME}/"

echo "Model"
echo "  - path: ${MODEL_DIR_NAME} (${MODEL_PATH})"
echo "  - epochs: ${EPOCHS}"
echo "  - task: ${TASK_NAME}"

if [[ ! -f "${MODEL_PATH}/model-${EPOCHS}.index" ]]; then
  python -u pix_lab/main/train_aru.py \
    --path_list "/home/vvasin/ikutukov/bd/train/????-20??/*.jp*g" \
    --output_folder "${MODEL_PATH}" \
    --restore_path "${MODEL_PATH}/" \
    --model_name "${MODEL_NAME}" \
    --cost_name "${COST_NAME}" \
    --optimizer_name "${OPTIMIZER_NAME}" \
    --queue_capacity 32 \
    --steps_per_epoch "${STEPS}" \
    --epochs ${EPOCHS} \
    --scale_min 0.33 \
    --scale_max 1.0 \
    --seed "${SEED}" \
    --max_pixels 4000000 \
    --gpu_device 0 \
    --image2tb_every_step 10
fi

# done
