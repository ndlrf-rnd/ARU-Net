set -ex

# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/vvasin/miniconda3/pkgs/cudatoolkit-10.0.130-0/lib"
for f in models/** ; do
    MODEL_NAME="$(basename ${f})"
    pb_path="./models/${MODEL_NAME}/model50.pb"
    if [[ -e "${pb_path}" ]] ; then
      python "run_demo_inference.py" \
        --path_list_imgs="./input/*.jpg" \
        --path_net_pb="${pb_path}" \
        --output_dir="./output/${MODEL_NAME}/"    
    fi
done

# python "run_demo_inference.py" \
#   --path_list_imgs="./input/*.jpg" \
#   --path_net_pb="demo_nets/model100_ema.pb" \
#   --output_dir="./output/aru-net-demo-100x256-ema/"

# python "run_demo_inference.py" \
#   --path_list_imgs="./input/*.jpg" \
#   --path_net_pb="models/aru-net-cbad-2019-300dpi-2560x10-ema.pb" \
#   --output_dir="./output/aru-net-cbad-2019-300dpi-2560x10-ema/"

# python "run_demo_inference.py" \
#   --path_list_imgs="./input/*.jpg" \
#   --path_net_pb="models/aru-net-cbad-2017-simple-300dpi-2560x10-ema.pb" \
#   --output_dir="./output/aru-net-cbad-2017-simple-300dpi-2560x10-ema/"


# python "run_demo_inference.py" \
#   --path_list_imgs="./input/*.jpg" \
#   --path_net_pb="models/aru-net-cbad-2017-complex-300dpi-2560x10-ema.pb" \
#   --output_dir="./output/aru-net-cbad-2017-complex-300dpi-2560x10-ema/"


# python "run_demo_inference.py" \
#   --path_list_imgs="./input/*.jpg" \
#   --path_net_pb="models/aru-net-bozen-2016-300dpi-2560x10-ema.pb" \
#   --output_dir="./output/aru-net-bozen-2016-300dpi-2560x10-ema/"


# python "run_demo_inference.py" \
#   --path_list_imgs="./input/*.jpg" \
#   --path_net_pb="models/aru-net-cbads-300dpi-100x256-ema.pb" \
#   --output_dir="./output/aru-net-cbads-300dpi-100x256-ema/"


# python "run_demo_inference.py" \
#   --path_list_imgs="./input/*.jpg" \
#   --path_net_pb="models/aru-net-cbads-200dpi-100x256-ema.pb" \
#   --output_dir="./output/aru-net-cbads-200dpi-100x256-ema/"

