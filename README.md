## INSTALLING PYTORCH AND NECESSARY PACKAGES
- First, create a virtual environment.
- Then, make sure that [CUDA](https://developer.nvidia.com/cuda-downloads) is installed (if you have an NVIDIA GPU), although this is not strictly necessary.
- After this, type `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` to install PyTorch.
- Then, run the following command: `python -m pip install matplotlib`.
- After this, all the necessary Python packages will be installed.


## DOWNLOADING THE DATA
- To download the required dataset for this project, go to [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) and download the zip file.
- Then, after installing all the necessary packages and creating your virtual environment, run `data_loader.py`. This will create all the necessary directories.
- Finally, extract the data from the zip file directly into the directory named `datasets`. If done correctly, `datasets` should contain two directories named `test` and `train`.