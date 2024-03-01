## Environment  
Windows 10 Pro, Ryzen 9 5900 12-Core Process, 3701Mhz, NVIDIA RTX 4070 TI  
CUDA 11.8  

## Start From a CommandLine (No Multicore for CPU)  
1. Start x64 Native Tools Command Prompt for VS 2022 for cl.exe environment
2. nvcc -c gpu_mul.cu -o gpu_mul.obj
3. nvcc -c main.cpp -o main.obj
4. nvcc -o gpu_mul.exe gpu_mul.obj main.obj
## Result  
cpu without parallel : 
gpu :  
