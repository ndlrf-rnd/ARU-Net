# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/vvasin/miniconda3/pkgs/cudatoolkit-10.0.130-0/lib"

for mp in `find models/** -type d` ; do

    MODEL_NAME="$(basename ${mp})"
    EXPORT_PATH="models/${MODEL_NAME}-ema.pb"

    if [[ ! -f "${EXPORT_PATH}" ]]; then
        echo "${MODEL_NAME} ---> ${EXPORT_PATH}"

        python "./pix_lab/main/export_ckpt.py" \
            "--restore_ckt_path=models/${MODEL_NAME}/model10" \
            "--export_name=${EXPORT_PATH}"

    else
        echo "${MODEL_NAME} -X-> ${EXPORT_PATH} - ALREADY EXISTS"
    fi

done