set -ex
export EPOCH="${EPOCH:-25}"

for f in models/** ; do
    MODEL_NAME="$(basename ${f})"
    PB_PATH="./models/${MODEL_NAME}/model${EPOCH}.pb"
    
    if [[ -e "${PB_PATH}" ]] ; then
      python "run_demo_inference.py" \
        --path_list_imgs="./input/*.jpg" \
        --path_net_pb="${PB_PATH}" \
        --output_dir="./output/${MODEL_NAME}/"    
    fi
done
