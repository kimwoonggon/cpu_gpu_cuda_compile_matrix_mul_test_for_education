## Start From a CommandLine
1. Start x64 Native Tools Command Prompt for VS 2022 for cl.exe environment
2. nvcc -c gpu_mul.cu -o gpu_mul.obj
3. nvcc -c main.cpp -o main.obj
4. nvcc -o gpu_mul.exe gpu_mul.obj main.obj
