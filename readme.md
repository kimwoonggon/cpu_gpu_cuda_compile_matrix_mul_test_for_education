## Purpose  
1. How much Can my cpu catch a speed of CUDA KERNEL which is not fully optimized just using global memory?
2. CPP 17 and CUDA nvcc Compiler can be integrated Naturally?  

## Environment  
Windows 10 Pro, Ryzen 9 5900 12-Core Process, 3701Mhz, NVIDIA RTX 4070 TI  
CUDA 11.8  

## Start From a CommandLine (No Multicore for CPU)  
1. Start x64 Native Tools Command Prompt for VS 2022 for cl.exe environment
2. nvcc -c gpu_mul.cu -o gpu_mul.obj
3. nvcc -c main.cpp -o main.obj
4. nvcc -o gpu_mul.exe gpu_mul.obj main.obj
## Result 1000x1000 Image Processing  
cpu without parallel :  856010 ms
cuda kernel gpu :  400.214 ms  

## OpenMP  
1. nvcc -c gpu_mul.cu -o gpu_mul.obj -O3
2. nvcc -c main.cpp -o main.obj -Xcompiler "/openmp,/O2"
3. nvcc -o gpu_mul.exe gpu_mul.obj main.obj
## Result 1000x1000 Image Processing  
cpu with naive : 62314 ms  
cpu with openmp :  8317 ms  
cuda kernel gpu :  374 ms  

## AVX2  
cl /EHsc /Ox /arch:AVX2 /openmp main_avx.cpp /Fe:mulavx.exe  
## Result 1000x1000 Image Processing (100x times)  
cpu with naive : 65422 ms  
cpu with avx2 :  48422 ms  
cpu with avx2+openmp : 6710 ms  



