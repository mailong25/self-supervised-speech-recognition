# Dependencies

Python:
 - python3
 - fairseq==0.9.0
 - torch >= 1.3.0
 - librosa
 - soundfile
 - h5py
 - pydub
 - sh
 
# [Wav2letter dependencies](https://github.com/facebookresearch/wav2letter/wiki/Dependencies)

# For GPU version:
Install NCCL (https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#down)
Install CUDNN (https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

Make sure you have cmake >= 3.15
Running the following as root

==================================================================
### [Array fire](https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux)
```
apt-get update
apt-get upgrade
apt-get install libboost-all-dev
apt-get install -y build-essential git cmake libfreeimage-dev
apt-get install -y cmake-curses-gui
apt-get install libopenblas-dev libfftw3-dev liblapacke-dev
apt-get install libatlas3gf-base libatlas-dev libfftw3-dev liblapacke-dev
git clone --recursive https://github.com/arrayfire/arrayfire.git --branch v3.6.4 && cd arrayfire
mkdir build && cd build
For CPU: cmake .. -DCMAKE_BUILD_TYPE=Release -DAF_BUILD_CUDA=OFF -DAF_BUILD_OPENCL=OFF
For GPU: cmake .. -DCMAKE_BUILD_TYPE=Release -DAF_BUILD_CUDA=ON
make -j8 && make install
cd ../..
```

==================================================================
### [Gloo](https://github.com/facebookincubator/gloo.git)
```
apt-get install openmpi-bin openmpi-common libopenmpi-dev
Download and unzip gloo at: (https://drive.google.com/file/d/1npI9V9ctXytrZFEIAmIRMwq5yPG-DXPh/view?usp=sharing)
cd gloo && mkdir -p build && cd build
cmake .. -DUSE_MPI=ON
make -j8 && make install
cd ../..
```

==================================================================
### [mkl-dnn](https://github.com/intel/mkl-dnn)

Please install [MKL](https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2018-install-guide) first if /opt/intel/mkl does not exists.
```
git clone https://github.com/intel/mkl-dnn.git -b mnt-v0
cd mkl-dnn && mkdir -p build && cd build
cmake .. 
make -j8 && make install
cd ../..
```

==================================================================
Install MKL:https://codeyarns.com/2019/05/14/how-to-install-intel-mkl/
==================================================================

==================================================================
### [flashlight](https://github.com/facebookresearch/flashlight.git)
```
export MKLROOT=/opt/intel/mkl
Download and unzip flashlight at: (https://drive.google.com/file/d/17MNp_pODBChVmX2sSUzHPkmPq856tzYl/view?usp=sharing)
cd flashlight && mkdir -p build && cd build
For CPU: cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CPU
For GPU: cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA
make -j8 && make install
cd ../..
```

==================================================================
### [kenlm](https://github.com/kpu/kenlm)
```
apt-get install libsndfile-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
apt-get install liblzma-dev libbz2-dev libzstd-dev
apt-get install libeigen3-dev
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir -p build && cd build
cmake .. -DKENLM_MAX_ORDER=20
make -j8 && make install
cd ../..
```

==================================================================
### [wav2letter with CPU backend](https://github.com/maltium/wav2letter/tree/feature/loading-from-hdf5)

Please change KENLM_ROOT_DIR=path/to/kenlm to your actual path to kenlm.
```
apt-get install libhdf5-dev
export MKLROOT=/opt/intel/mkl && export KENLM_ROOT_DIR=path/to/kenlm
git clone https://github.com/mailong25/wav2letter.git
cd wav2letter && mkdir -p build

For CPU: 
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DW2L_LIBRARIES_USE_CUDA=OFF -DKENLM_MAX_ORDER=20
make -j8

For GPU: open file Train.cpp and replace the line:
reducer = std::make_shared<fl::InlineReducer>(1.0 / fl::getWorldSize()); by the line: 
reducer = std::make_shared<fl::CoalescingReducer>(1.0 / fl::getWorldSize(),true,true);
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=20 -DW2L_LIBRARIES_USE_CUDA=ON
make -j8
```
