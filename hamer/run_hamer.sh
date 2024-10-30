#!/bin/bash
#SBATCH -p rtx3090_slab
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 16

# 设置ulimit防止内存溢出
ulimit -s unlimited

# 激活conda环境
source ~/.bashrc
conda activate hamer2

# 设置CUDA环境
export CUDA_HOME=/mnt/slurm_home/share/cuda/11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置 PyRender 相关环境变量
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

# 设置Python内存垃圾回收
export PYTHONMALLOC=debug
export MALLOC_TRIM_THRESHOLD_=65536

# 检查参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <video_path> <output_dir>"
    exit 1
fi

VIDEO_PATH="$1"
OUTPUT_DIR="$2"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行程序，添加错误处理
python demo.py \
    --v_path "$VIDEO_PATH" \
    --out_folder "$OUTPUT_DIR" \
    --batch_size=16 \
    --side_view \
    --save_mesh \
    --full_frame \
    2>&1 | tee "$OUTPUT_DIR/run.log"

# 检查运行状态
if [ $? -eq 0 ]; then
    touch "${OUTPUT_DIR}/.processed"
    echo "Processing completed successfully"
else
    echo "Processing failed with error code $?"
    exit 1
fi