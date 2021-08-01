# Install tensorflow
pip install tensorflow==2.5.0

rm -rf ./tensorflow-large-model-support
git clone --branch tflmsv2 https://github.com/IBM/tensorflow-large-model-support.git
pip install ./tensorflow-large-model-support
pip install memory_profiler


conda install -y -c conda-forge gxx_linux-64
conda uninstall -y --force jpeg libtiff
conda install -y -c conda-forge libjpeg-turbo --no-deps
pip uninstall -y pillow

sudo yum install -y zlib-devel libtiff-devel libjpeg-dev

export CXX=x86_64-conda-linux-gnu-g++
export CC=x86_64-conda-linux-gnu-gcc

# PY 3.7
# CFLAGS="$CFLAGS -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd

# # PY 2.7
CFLAGS="$CFLAGS -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd==v6.2.2.post1

## Make sure Pillow doesn't get reinstalled
python -c 'import PIL.features; print(PIL.features.check_feature("libjpeg_turbo"))'


## Instlal lib
conda install --yes click tqdm
pip install scikit-image scipy==1.1.0