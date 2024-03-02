# CUDA and CPU Matrix Multiplication Performance Comparison
These code snippets present a comparison of matrix multiplication performance using a CPU with various optimizations (including OpenMP and AVX2) and a GPU using CUDA.  
The aim is to demonstrate the potential speedup achievable with CUDA for tasks not fully optimized and relying only on global memory, compared to different CPU optimization techniques.  

  
## Environment Setup
Operating System: Windows 10 Pro  
CPU: Ryzen 9 5900, 12-Core Processor, 3.701 GHz  
GPU: NVIDIA RTX 4070 TI  
CUDA Version: 11.8  
Development Tools: x64 Native Tools Command Prompt for VS 2022, CUDA nvcc Compiler  
  
## Compilation Steps  
To compile the code, follow these steps from the Command Line (Note: No multicore processing for CPU):  
- Build and Compile Using CMake:  
```
mkdir build
cd build
cmake -S .. -B .
cmake --build . --config Release
```  
## Performance Results  
Performance was tested on 1000x1000 matrix multiplication tasks:  

- Without Parallelization:  
CPU: 856,010 ms  
CUDA GPU: 400.214 ms  
- With OpenMP:  
CPU: 8,317 ms  
CUDA GPU: 374 ms  
- With AVX2:  
CPU (naive): 65,422 ms  
CPU with AVX2: 48,422 ms  
CPU with AVX2 + OpenMP: 6,710 ms

## Additional Information  
The copyright of the provided code and methodology belongs to Woonggon Kim. Interested parties are encouraged to contact for further information or collaboration opportunities.



