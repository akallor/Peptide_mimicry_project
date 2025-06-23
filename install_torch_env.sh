#The TPU torch incompatiblity is preventing torch from being installed, so we are making
#a new conda environment for Python 3.10
#We need to do this to be able to properly compute the embedding network using ESM models (any)
# Install Miniconda
!wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash miniconda.sh -b -p /opt/conda
!rm miniconda.sh

# Add conda to PATH
import sys
sys.path.append('/opt/conda/bin')

import os
os.environ['PATH'] = '/opt/conda/bin:' + os.environ['PATH']

!conda create -y -n esm-tpu python=3.10
!conda run -n esm-tpu python -m pip install --upgrade pip
!conda run -n esm-tpu pip install torch==2.0.0
!conda run -n esm-tpu pip install https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
!conda run -n esm-tpu pip install fair-esm pandas tqdm
