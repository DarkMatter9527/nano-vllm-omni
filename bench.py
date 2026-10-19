import statistics
import time

from PIL import Image

from nanovllm_omni import DEFAULT_IMAGE, DEFAULT_PROMPT, OmniLLM


# 测试主逻辑
def main():
    # 初始化引擎：加载模型权重并构建调度器/执行器
    llm = OmniLLM()
    # 使用内置 quality 预设的采样参数（避免就地修改污染预设）
    sampling_params = llm.preset("quality")
    # 读取默认示例图像并转成 RGB
    image = Image.open(DEFAULT_IMAGE).convert("RGB")

    # Warm up once after model load, following nano-vllm's bench style.
    # 预热一次：完成 CUDA 初始化、模块 offload 换入换出等一次性开销，不计入正式计时
    warmup_output = llm.generate(image=image, prompt=DEFAULT_PROMPT, sampling_params=sampling_params)
    print(
        f"warmup: frames={len(warmup_output.frames)} size={warmup_output.width}x{warmup_output.height} "
        f"fps={warmup_output.fps}"
    )

    times = []
    # 连续计时 5 次，只统计 generate 耗时（不含模型加载）
    for run_idx in range(5):
        start = time.perf_counter()
        output = llm.generate(image=image, prompt=DEFAULT_PROMPT, sampling_params=sampling_params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(
            f"run={run_idx + 1} time_s={elapsed:.4f} frames={len(output.frames)} "
            f"size={output.width}x{output.height}"
        )

    # 汇总统计：均值 / 最小 / 最大，以及逐次耗时列表
    print("summary")
    print(f"mean_s={statistics.mean(times):.4f}")
    print(f"min_s={min(times):.4f}")
    print(f"max_s={max(times):.4f}")
    print("times_s=" + ",".join(f"{t:.4f}" for t in times))


if __name__ == "__main__":
    main()
