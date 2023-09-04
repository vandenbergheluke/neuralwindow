# neuralwindow

## Setup

Clone the repository

```bash
git clone https://github.com/vandluke/neuralwindow.git
```

Make project the current working directory

```bash
cd neuralwindow 
```

### Ubuntu 22.04 LTS

Update and upgrade Ubuntu

```bash
sudo apt update && sudo apt upgrade
```

Install MKL

```bash
sudo apt-get install -y intel-mkl
```

Install OpenBLAS

```bash
sudo apt-get install -y libopenblas-dev
```

Install Check

```bash
sudo apt-get install -y check
```

Install Valgrind

```bash
sudo apt-get install -y valgrind
```

Install Libtorch

```bash
sudo wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-shared-with-deps-2.0.1+cpu.zip
```

Install CUDA

```bash
lspci | grep -i nvidia  
sudo apt-get install linux-headers-$(uname -r)  
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get -y install cuda
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
```

After CUDA installation you will need to reboot. Verify installation with

```bash
nvcc --version
```

## Build

To build neuralwindow run

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch/ ..
make
```

To build without GPU define enviroment variable `CPU_ONLY=1`.

## Test

Run all the unit tests with

```bash
make test
```

A specific test executable can be run using the command

```bash
make <test_name>
```

Valgrind can be run with a specific test exectuable with the command

```bash
make valgrind_<test_name>
```

To generate a Valgrind suppression file run

```bash
valgrind --leak-check=full --show-reachable=yes --error-limit=no --gen-suppressions=all --log-file=suppressions.log ./test/<test_name>
cat ./suppressions.log | ./../parse_valgrind_suppressions.sh > suppressions.supp
```

To test without GPU define enviroment variable `CPU_ONLY=1`.
