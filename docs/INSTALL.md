# Installation

```bash
# install libraries
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html

pip3 install -r requirements.txt

sudo apt-get update & sudo apt-get install ffmpeg libsm6 libxext6  -y

# detectron2==0.6 build from source
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

```