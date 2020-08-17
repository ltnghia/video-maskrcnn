## Installation

### Requirements:
- PyTorch 1.2.0. Installation instructions can be found in https://pytorch.org/get-started/locally/
- CUDA 10.1
- cocoapi
- GCC >= 4.9
- OpenCV

### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name video-maskrcnn
conda activate video-maskrcnn

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1
conda install -c pytorch pytorch torchvision cudatoolkit=10.1

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/ltnghia/video-maskrcnn.git
cd video-maskrcnn

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

### Option 2: Docker Image (Requires CUDA, Linux only)

Build image with defaults (`CUDA=10.1`, `CUDNN=7`, `FORCE_CUDA=1`):

    nvidia-docker build -t video-maskrcnn docker/
    
Build image with other CUDA and CUDNN versions:

    nvidia-docker build -t video-maskrcnn --build-arg CUDA=10.1 --build-arg CUDNN=7 docker/
    
Build image with FORCE_CUDA disabled:

    nvidia-docker build -t video-maskrcnn --build-arg FORCE_CUDA=0 docker/
    
Build and run image with built-in jupyter notebook(note that the password is used to log in jupyter notebook):

    nvidia-docker build -t video-maskrcnn-jupyter docker/docker-jupyter/
    nvidia-docker run -td -p 8888:8888 -e PASSWORD=<password> -v <host-dir>:<container-dir> video-maskrcnn-jupyter
