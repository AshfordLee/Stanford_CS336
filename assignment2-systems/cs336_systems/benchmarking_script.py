import torch
import argparse
import cs336_systems
import cs336_basics
import cs336_basics.transformer as transformer
import timeit
import numpy as np
import json
import os
import pandas as pd
from loguru import logger

def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description="Time & Memory test for Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model Config
    model_group = parser.add_argument_group('model config')

    model_group.add_argument(
        "--model-size",  # 添加 -- 前缀
        type=str,
        default="small",
        choices=["tiny","small","medium","large"],
        help="Size of model"
    )

    model_group.add_argument(
        "--num-layers",  # 添加 -- 前缀
        type=int,
        default=4,
        help="Num of layers in Transformer model"
    )

    model_group.add_argument(
        "--num-heads",  # 添加 -- 前缀
        type=int,
        default=16,
        help="Num of heads in Transformer model"
    )

    model_group.add_argument(
        "--d-model",  # 添加 -- 前缀
        type=int,
        default=512,
        help="d_model of Transformer model"
    )

    model_group.add_argument(
        "--d-ff",  # 添加 -- 前缀
        type=int,
        default=1344,
        help="d_ff(forward) of Transformer model"
    )

    model_group.add_argument(
        "--vocab-size",  # 添加 -- 前缀
        type=int,
        default=10000,
        help="vocab_size of Transformer model"
    )

    model_group.add_argument(
        "--context-length",  # 添加 -- 前缀
        type=int,
        default=256,
        help="context_length of Transformer model"
    )

    model_group.add_argument(
        "--use-rope",  # 添加 -- 前缀，修正为 store_true
        action="store_true",
        default=True,
        help="use_rope or not"
    )

    model_group.add_argument(
        "--max-seq-len",  # 添加 -- 前缀
        type=int,
        default=256,
        help="max sequence length of Transformer model"
    )

    model_group.add_argument(
        "--theta",  # 添加 -- 前缀
        type=int,
        default=10000,
        help="theta of Transformer model"
    )

    model_group.add_argument(
        "--num-warmup",  # 添加 -- 前缀
        type=int,
        default=5,
        help="num of warm_up steps"
    )

    model_group.add_argument(
        "--num-iterations",  # 添加 -- 前缀
        type=int,
        default=100,
        help="num of iterations in benchmark test"
    )

    # 添加缺失的参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for benchmarking"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )

    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="Include backward pass (gradient computation)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )

    return parser.parse_args()

def create_model(args):
    model = transformer.transformer_lm(
        d_model = args.d_model,
        num_heads=args.num_heads, 
        d_ff=args.d_ff,
        vocab_size=args.vocab_size, 
        context_length=args.context_length,
        num_layers=args.num_layers, 
        use_rope=args.use_rope,
        max_seq_len=args.max_seq_len, 
        theta=args.theta
    )

    model.to(args.device)

    return model

def generate_random_batch(vocab_size, context_length, batch_size, device):

    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    return input_ids

def run_forward_backward(model, batch, include_backward=False):
    outputs = model(batch)

    if include_backward:
        # 创建虚拟loss并反向传播
        loss = outputs.sum()  # 简化的loss
        loss.backward()

    return outputs


def benchmark_model(model, args):

    batch = generate_random_batch(
        args.vocab_size, 
        args.context_length, 
        args.batch_size, 
        args.device
    )

    logger.info(f"Running {args.num_warmup} steps of warming:")
    model.train() if args.include_backward else model.eval()

    for _ in range(args.num_warmup):
        run_forward_backward(model, batch, args.include_backward)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    logger.info(f"Running {args.num_iterations} steps:")
    times = []

    for _ in range(args.num_iterations):
        # 清除之前的梯度
        if args.include_backward:
            model.zero_grad()
        
        # 计时开始
        start_time = timeit.default_timer()
        
        # 执行一步
        run_forward_backward(model, batch, args.include_backward)
        
        # GPU同步
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时结束
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    
    return times

def analyze_and_save_results(times, args, save=False):
    import numpy as np
    import json
    import os
    
    # 统计信息
    times_array = np.array(times)
    
    # 构建结果字典
    results = {
        'statistics': {
            'mean_time': float(times_array.mean()),
            'std_time': float(times_array.std()),
            'min_time': float(times_array.min()),
            'max_time': float(times_array.max()),
            'median_time': float(np.median(times_array)),
        },
        'config': {
            'model_size': args.model_size,
            'num_layers': args.num_layers,
            'context_length': args.context_length,
            'batch_size': args.batch_size,
            'device': args.device,
            'include_backward': args.include_backward,
            'num_iterations': args.num_iterations
        }
    }

    # === 简洁输出格式 ===
    print("\nBenchmark Results")
    print("=" * 50)
    
    # 配置信息
    config = results['config']
    print("Configuration:")
    print(f"  Model: {config['model_size']}, Layers: {config['num_layers']}, Context: {config['context_length']}")
    print(f"  Batch size: {config['batch_size']}, Device: {config['device']}, Backward: {config['include_backward']}")
    print(f"  Iterations: {config['num_iterations']}")
    
    # 统计结果
    stats = results['statistics']
    print("\nStatistics (seconds):")
    print(f"  Mean time:    {stats['mean_time']:.4f}")
    print(f"  Std time:     {stats['std_time']:.4f}")
    print(f"  Min time:     {stats['min_time']:.4f}")
    print(f"  Max time:     {stats['max_time']:.4f}")
    print(f"  Median time:  {stats['median_time']:.4f}")
    
    print("=" * 50)

    if save:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 生成实验名称
        precision = getattr(args, 'precision', 'fp32')
        exp_name = args.experiment_name or f"{args.model_size}_{args.context_length}_{args.batch_size}_{precision}"
        output_file = os.path.join(args.output_dir, f"{exp_name}_results.json")
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
    
    return results
    

def main():
    args = parse_arguments()
    logger.info(f"Running on device:{args.device}")

    torch.manual_seed(args.seed)

    logger.info(f"Creating models:")

    model = create_model(args)

    logger.info(f"Running tests:")

    times = benchmark_model(model, args)

    analyze_and_save_results(times, args)

    

if __name__ == "__main__":
    main()




