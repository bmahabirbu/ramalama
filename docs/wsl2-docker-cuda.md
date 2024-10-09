# Setting Up Ramalama with CUDA Support in WSL2 Using Podman


This document outlines the steps required to get Ramalama up and running in WSL2 with CUDA support using Docker.

## Prerequisites
1. **Install Game-Ready Drivers on Windows**  
   Ensure that you have the appropriate NVIDIA game-ready drivers installed on your Windows system. This is necessary for CUDA support in WSL2.
2. **Install Docker Desktop**
  In settings under general check :white_check_mark: Use the WSL 2 based engine

## Installing CUDA Toolkit
1. **Install the CUDA Toolkit**  
   Follow the instructions in the [NVIDIA CUDA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) to install the CUDA toolkit.

   - **Download the CUDA Toolkit:**  
     Visit the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local).

2. **First Run the Following Command From the WSL User Guide:**
   ```bash
   sudo apt-key del 7fa2af80
   ```

3. **Then Select Your Environment and Download Cuda Toolkit:**  
   Go back to the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) and select your appropriate environment. Install the package using the deb format or your preferred method.

   > **Note:** This package enables WSL to interact with Windows drivers, allowing CUDA support.

4. **Install NVIDIA Container Toolkit (optional)**  
   Follow the installation instructions provided in the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

5. **Run the Following Command:**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```
   - **Note:** No need to restart Docker in Linux, as it's running in Windows. Restarting WSL should apply the changes.

   > **Important:** This package is essential for Docker to access CUDA for building images (not needed for running with gpu).

## Testing the Setup

1. **Run the following command to test the installation:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.1-devel-ubi9 nvidia-smi
   ```

2. **Expected Output:**  
   The output should resemble the following:

   ```
   Mon Dec 27 02:18:10 2021
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 495.53       Driver Version: 497.29       CUDA Version: 11.5     |
   |-------------------------------+----------------------+----------------------|
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
   |  0%   47C    P0    23W / 100W |   1781MiB /  4096MiB |     N/A      Default |
   |                               |                      |                  N/A |
   +-------------------------------+----------------------+----------------------|

   +-----------------------------------------------------------------------------+
   | Processes:                                                                  |
   |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
   |        ID   ID                                                   Usage      |
   |=============================================================================|
   |  No running processes found                                                 |
   +-----------------------------------------------------------------------------+
   ```