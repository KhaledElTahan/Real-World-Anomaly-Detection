# Install NVIDIA driver
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo apt-get update
sudo ubuntu-drivers autoinstall
# Reboot. 

# Check that GPUs are visible
nvidia-smi

# Update apt-get
sudo apt-get update

# Download Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh

# Install Anaconda
bash Anaconda3-2021.05-Linux-x86_64.sh

# Add Anaconda to $PATH
echo "export PATH=~/anaconda3/bin:$PATH" >> /.bashrc

# Reload environment variables
source ~/.bashrc

# Install CUDA
conda install -c anaconda cudatoolkit 
sudo apt install nvidia-cuda-toolkit

# Check CUDA installation
nvcc --version

# Install Cudnn https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

sudo apt-get install libcudnn8=8.2.4.*-1+cuda11.4
sudo apt-get install libcudnn8-dev=8.2.4.*-1+cuda11.4

# Install Sckikit-Learn
conda install -c anaconda scikit-learn 

# Install PyTorch
conda install -c pytorch pytorch

# Test PyTorch & GPU Installation
python << END
import torch
print(torch.cuda.is_available())
END

# Install torchvision
conda install -c pytorch torchvision

# Install ffmpeg
conda install -c conda-forge ffmpeg 

# Install fvcore
conda install -c conda-forge fvcore 

# Install iopath
conda install -c iopath iopath

# Install pytorch video
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"

# Install Open-CV
pip install opencv-python

# Test Open-CV
python << END
import cv2
print(cv2.__version__)
END

# Install SimpleJson
conda install -c conda-forge simplejson

# Install AV
conda install -c conda-forge av

# Install Update YACS, since the conda version is outdated
pip install yacs -U

# If run_net.py prints error "ImportError: libopenh264.so.5: cannot open shared object file: No such file or directory"
# based on this https://stackoverflow.com/questions/62213783/ffmpeg-error-while-loading-shared-libraries-libopenh264-so-5
ln -s ~/anaconda3/lib/libopenh264.so ~/anaconda3/lib/libopenh264.so.5

