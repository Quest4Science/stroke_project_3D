import os
import glob
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configuration
INPUT_DIR = "/mnt/slurm_home/htli/program/Data/hand_to_table/original_001"
OUTPUT_BASE_DIR = "/mnt/slurm_home/htli/program/Data/hand_to_table/hamer_output_001"
PARTITION = "rtx3090_slab"
MAX_CONCURRENT_JOBS = 12
SLEEP_TIME = 60  # Time to wait between checks (in seconds)

# Set up logging
log_file = f"job_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def get_available_nodes():
    """Get the number of available nodes in the specified partition."""
    cmd = f"sinfo -p {PARTITION} -h -t idle,mix -o '%n'"
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return len(result.stdout.strip().split())
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting available nodes: {e.stderr.strip()}")
        return 0


def get_running_jobs():
    """Get the number of currently running jobs."""
    try:
        result = subprocess.run(
            "squeue -h -u $USER -p rtx3090_slab | wc -l",
            shell=True, stdout=subprocess.PIPE, text=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting running jobs: {e}")
        return MAX_CONCURRENT_JOBS  # Assume max jobs to prevent further submissions


def submit_job(video_path, output_dir):
    """Submit a job using nohup and srun."""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否已经处理过
    if os.path.exists(os.path.join(output_dir, ".processed")):
        logging.info(f"Video {video_path} has already been processed, skipping...")
        return True

    # 生成作业名称
    job_name = f"hamer_{Path(video_path).stem}"

    # 构建运行命令
    srun_cmd = (
        f"srun -p {PARTITION} -n 1 --job-name={job_name} "
        f"--gres=gpu:1 --kill-on-bad-exit=1 -c 16 "
        f"python demo.py "
        f"--v_path {video_path} "
        f"--out_folder {output_dir} "
        f"--batch_size=48 --side_view --save_mesh --full_frame"
    )

    # 构建完整的nohup命令
    log_file = os.path.join(output_dir, f"{job_name}.log")
    cmd = f"nohup {srun_cmd} > {log_file} 2>&1 &"

    try:
        subprocess.run(cmd, shell=True, check=True)
        logging.info(f"Successfully submitted job for {job_name} in background")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job for {job_name}. Error: {e}")
        return False


def check_job_completion(output_dir):
    """Check if a job has completed by looking for the .processed file."""
    return os.path.exists(os.path.join(output_dir, ".processed"))


def main():
    # 获取所有视频文件
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mkv"))
    total_files = len(video_files)
    processed_files = 0
    running_jobs = {}  # 记录正在运行的作业

    logging.info(f"Starting job submission for {total_files} video files")

    while processed_files < total_files or running_jobs:
        current_running = get_running_jobs()
        available_nodes = get_available_nodes()

        # 检查正在运行的作业的状态
        for video_path, output_dir in list(running_jobs.items()):
            if check_job_completion(output_dir):
                logging.info(f"Job completed for {video_path}")
                del running_jobs[video_path]
                processed_files += 1
                logging.info(f"Progress: {processed_files}/{total_files} files processed")

        # 提交新作业
        if current_running < MAX_CONCURRENT_JOBS and available_nodes > 0:
            for video_path in video_files:
                if video_path not in running_jobs and not check_job_completion(
                        os.path.join(OUTPUT_BASE_DIR, Path(video_path).stem)
                ):
                    output_dir = os.path.join(OUTPUT_BASE_DIR, Path(video_path).stem)
                    if submit_job(video_path, output_dir):
                        running_jobs[video_path] = output_dir
                        time.sleep(2)  # 短暂延迟，避免提交过快
                        break

        # 等待一段时间后再检查
        logging.info(f"Status: Running jobs: {current_running}, "
                     f"Available nodes: {available_nodes}, "
                     f"Processed: {processed_files}/{total_files}")
        time.sleep(SLEEP_TIME)

    logging.info("All jobs completed successfully")


if __name__ == "__main__":
    start_time = datetime.now()
    logging.info(f"Script started at {start_time}")

    try:
        main()
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        end_time = datetime.now()
        logging.info(f"Script ended at {end_time}")
        logging.info(f"Total runtime: {end_time - start_time}")