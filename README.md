# simnets-tf
## SimNets implementation in TensorFlow

### Binary installation
Binary installation requires a cuda toolit installation >= 7.5. <BR/>
Download the .whl file from the GitHub release tab, then type:
```
python -m pip install <whl file>
```
all requirements should be installed automatically.

### Building from Source
Building from source requires:
1. A working c++ compiler with c++11 support (gcc >= 4.7)
2. Cuda toolkit installed (for nvcc)
3. CMake >= 3.0 (<code>apt install cmake</code>)
4. TensorFlow installed for the Python interpreter you intend to use

<B>Important:</B> The following command should run without error:
```
python -c 'import tensorflow as tf'
```
To build the project type the following commands:<BR/>
 Python 2.7:<BR/>
 ```
 git clone https://github.com/HUJI-Deep/simnets-tf.git
 cd simnets-tf
 mkdir build
 cd build
 cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMNETS_PYTHON_VERSION=2.7 -DCMAKE_INSTALL_PREFIX=install
 make -j simnet_ops
 ```
 
 Python 3.5:<BR/>
  ```
  git clone https://github.com/HUJI-Deep/simnets-tf.git
  cd simnets-tf
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMNETS_PYTHON_VERSION=3.5 -DCMAKE_INSTALL_PREFIX=install
  make -j simnet_ops
  ```
 To test the code you can now type:
 ```
 make test_simnet_ops
 ```
 This should run for about two minutes and return without any errors.<BR/>
 Now you can create a .whl file:
 ```
 make create_wheel
 ```
 
 Fianlly, to install the simnets-tf package type (remember to use the right interpreter):
 ```
 cd install/dist
 python -m pip install <whl file>
 ```
 The installation is successful if the following runs (again, remember to use the right interpreter):
 ```
 python -c 'import simnets'
 ```
 
