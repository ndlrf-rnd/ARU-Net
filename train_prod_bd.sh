set -ex

export STEPS="${STEPS:-256}"
export EPOCHS="${EPOCHS:-200}"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/miniconda3/lib/"

export MODEL_NAME='aru'
export COST_NAME='cross_entropy'
export OPTIMIZER_NAME='rmsprop'
export TASK_NAME='bd'
export SEED=${SEED:-13}

export PYTHONPATH="$PYTHONPATH:~/ARU-Net"


function train_fn () {
    MODEL_PATH=$1
    MAX_PIXELS=${2:-4194304}

    echo "Model"
    echo "  - path: (${MODEL_PATH})"
    echo "  - epochs: ${EPOCHS}"
    echo "  - task: ${TASK_NAME}"

    if [[ ! -f "${MODEL_PATH}/model-${EPOCHS}.pb" ]]; then
      python -u pix_lab/main/train_aru.py \
        --path_list "../../DATA/${TASK_NAME}/train/????-20??/*.jp*g" \
        --output_folder "${MODEL_PATH}" \
        --restore_path "${MODEL_PATH}/" \
        --model_name "${MODEL_NAME}" \
        --cost_name "${COST_NAME}" \
        --optimizer_name "${OPTIMIZER_NAME}" \
        --classes 2 \
        --queue_capacity 16 \
        --steps_per_epoch "${STEPS}" \
        --epochs ${EPOCHS} \
        --scale_min 0.33 \
        --scale_max 1.0 \
        --seed "${SEED}" \
        --max_pixels ${MAX_PIXELS} \
        --gpu_device 0 \
        --checkpoint_every_epoch 5 \
        --sample_every_steps 32
    fi
}

train_fn "./models/${MODEL_NAME}-${COST_NAME}-${OPTIMIZER_NAME}-${TASK_NAME}-prod-8mpx/" 8388608 

train_fn "./models/${MODEL_NAME}-${COST_NAME}-${OPTIMIZER_NAME}-${TASK_NAME}-prod-4mpx/" 4194304 

