# How to install tensorflow-gpu2.0
* This tutorial only tests in the following list environment, and it is for `tensorflow-gpu 2.0` version.  
1. Windows10
2. NVIDIA GTX1080ti,GTX2080ti  
## Step 1: Download and install CUDA and cuDNN
### TF2 Only  
* If you just want to create a `tensorflow-gpu 2.0` environment and do not care other python environment, 
you can download the two files from NVIDIA official website.Must note that TF2 only work with CUDA>=10 
and cuDNN>=7.5.  
1. [CUDA](https://developer.nvidia.com/cuda-downloads)
2. [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)  
* After downloading, you can follow the traditional methods to install them.  
### Old python with TF2
* If you want to keep your old environment and create new `tensorflow-gpu 2.0` environment, you must follow the 
following steps. First download the two files through my Cloud disk.There are `.bz` files and can be install 
by `pip`. You can download from other sources if find the same files.  
1. [CUDA](https://pan.baidu.com/s/1cm-qfO9hzWDQpez9R7C43Q) (Extracted code:csce)
2. [cuDNN](https://pan.baidu.com/s/1gb-WpSpXgkAn57fA2n_P5Q) (Extracted code:p4ya)  
(If invalid please contact me with e-mail feixue@nuaa.edu.cn)  
After downloading, you get two files end with `.bz2`. Do not hurry to install. Here comes the key steps.  
## Step 2: Creat TF2 environment  
* `TF2 Only` can skip this step.  
* To keep the old environment safe, use the following methods. 
1. Install Miniconda:  
Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install normally. 
2. Use conda creating independent environment:  
Open conda in command line and create a environment must with python>=3.6  
```bash
conda create -n tf2 python=3.6
```  
3. Install CUDA and cuDNN  
Activate the TF2 environment and install everything  
```bash
conda activate tf2 
pip install [path to cudatoolkit-10.0.130-0.tar.bz2]  
pip install [path to cudnn-7.6.0-cuda10.0_0.tar.bz2]
```
## Step 3: Install TF2  
* After install CUDA and cuDNN, TF2 can be easily install  
```bash
pip install tensorflow-gpu==2.0.0b1
```  
* If you do not know which version of TF2 you can chose, just type in  
```bash
pip install tensorflow-gpu==2.0000000
```  
And pip will list candidates. This method works well on other python packages.    
* Finally you can start your TF2 python and have a test.  
```python
import tensorflow as tf
```  
* Installation is completed if there is nothing happened. Or you need to install some dependency packages.   
## Finally  
* If you want to use the old python, just start it in system command line.  
* If you want to use the TF2 python, you need to activate the TF2 conda environment then start python. 
## Recommended
* Pycharm has advantages  
1. Easy to chose python environment  
2. Quick open `tensorboard` (Add tensorboard into external tools. Open tensorboard with external tools.)  
* Tensorlayer is easy to use  
Tensorlayer is based on tensorflow, it works easily with  partial PyTorch syntax.  