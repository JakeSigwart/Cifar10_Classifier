# Cifar10_Classifier
Use Tensorflow to train a classifier on the Cifar10 dataset. This model was trained on an NVIDIA GEFORCE GTX 1060 GPU on a Windows machine.

## Getting Started
With a Python 3.6.5, pip install Numpy, Pillow, Matplotlib, Struct and Pickle. Most of these are included with standard python installations. Use the following command to install the latest version of Tensorflow:
```
C:\> pip install tensorflow
```
### Tensorflow for GPU
If you plan to run Tensorflow on a GPU, check hardware requirements at: https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support
```
C:\> pip install tensorflow-gpu
```
### Installing CUDA
Download CUDA toolkit from: https://developer.nvidia.com/cuda-downloads. Download cuDNN from: https://developer.nvidia.com/cudnn (Note: this requires an account). Add the locations of these programs to your 'Path' Environment variable.
### Cifar10 Dataset
Download this repository. Download the Cifar10 dataset and un-zip into the repository folder. The dataset is available at: https://www.cs.toronto.edu/~kriz/cifar.html.

## Performance
The model was trained on 900 batches of 256 images. This is a total of 230,400 images however, a pre-processing layer randomly distorted the images during training. The distortions involved changing contrast, hue and randomly flipping images horizontally. The optimization function used is AdamOptimizer with learning_rate=1e-4. The training accuracy is shown below.
![alt text](




