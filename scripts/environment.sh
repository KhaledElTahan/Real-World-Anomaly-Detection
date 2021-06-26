# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-450
# Reboot. 

# Check that GPUs are visible
nvidia-smi

# Update apt-get
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.5.32-1+cuda10.1  \
    libcudnn7-dev=7.6.5.32-1+cuda10.1

# Download Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh

# Install Anaconda
bash Anaconda3-2020.07-Linux-x86_64.sh

# Add Anaconda to $PATH
echo "export PATH=~/anaconda3/bin:$PATH" >> ~/.bashrc

# Reload environment variables
source ~/.bashrc

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
pip install pytorchvideo

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

# If run_net.py prints error "ImportError: libopenh264.so.5: cannot open shared object file: No such file or directory"
# based on this https://stackoverflow.com/questions/62213783/ffmpeg-error-while-loading-shared-libraries-libopenh264-so-5
ln -s ~/anaconda3/lib/libopenh264.so ~/anaconda3/lib/libopenh264.so.5

