# Problem (batch_size): Tune the batch size (3 points) (4 H100 hrs)

from cs336_basics import transformer
from loguru import logger
import numpy as np
import torch
import argparse
import json
import os
from datetime import datetime

logger.add('./../../Logs/Assignment1.log')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 数据路径
    parser.add_argument('--train_path', type=str, default='./../TinyStories_Result/train_tokens.npy')
    parser.add_argument('--val_path', type=str, default='./../TinyStories_Result/valid_tokens.npy')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pt')
    
    # 模型超参数 (已给出)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--use_rope', action='store_true', default=True)
    parser.add_argument('--max_seq_len', type=int, default=256)

    parser.add_argument('--alpha_min', type=float, default=1e-5, help='Minimum learning rate for cosine decay')
    
    # Batch size 搜索参数
    parser.add_argument('--batch_sizes', type=int, nargs='+', 
                       default=[64, 96, 128],
                       help='List of batch sizes to sweep')
    parser.add_argument('--results_dir', type=str, default='./../../HyperParameter_Result',
                       help='Directory to save sweep results')
    
    # 训练超参数 (使用默认值)
    parser.add_argument('--batch_size', type=int, default=64)  # 单个训练时的默认值
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=20000)       # 总训练步数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 控制参数
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    
    return parser.parse_args()

def log_metrics(step, train_loss, val_loss, current_lr=None, batch_size_value=None):
    if current_lr is not None:
        logger.info(f"Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")
    else:
        logger.info(f"Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # 返回用于记录学习曲线的数据
    return {
        'step': step,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'batch_size': batch_size_value if batch_size_value is not None else 64
    }

def create_model(args):
    d_model,num_heads,d_ff,vocab_size,context_length,num_layers,use_rope,max_seq_len,theta = args
    model = transformer.transformer_lm(d_model,num_heads,d_ff,vocab_size,context_length,num_layers,use_rope,max_seq_len,theta)
    return model

def create_optimizer(model,args):
    params, lr, betas, eps, weight_decay = args
    optimizer = transformer.AdamW(params,lr,betas,eps,weight_decay)
    return optimizer

def load_data_memmap(data_path):
    try:
        data = np.load(data_path,mmap_mode='r')
    except:
        data = np.memmap(data_path,dtype = np.int32, mode = 'r')
    return data

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size,seq_len,vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  
    targets_flat = targets.view(-1)  
    loss = transformer.cross_entropy(logits_flat, targets_flat) 
    return loss

@torch.no_grad()
def evaluate(model, val_data, batch_size=32, context_length=256, device='cpu'):
    model.eval()
    total_loss = 0
    num_batches = 0

    # 只评估前几个批次以加快速度
    max_eval_batches = 10  # 最多评估10个批次

    for i in range(max_eval_batches):
        x, y = transformer.data_loading(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = compute_loss(logits, y)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')

def train_single_batch_size(batch_size_value, args, train_data, val_data):
    """训练单个 batch size 配置"""
    logger.info(f"=== 开始 batch_size {batch_size_value} 的训练 ===")
    
    # 创建模型
    model = create_model((
        args.d_model, args.num_heads, args.d_ff, args.vocab_size,
        args.context_length, args.num_layers, args.use_rope,
        args.max_seq_len, args.theta
    ))
    model.to(args.device)
    
    # 创建优化器
    optimizer = create_optimizer(model, (
        model.parameters(), args.learning_rate,
        (args.beta1, args.beta2), args.eps, args.weight_decay
    ))
    
    # 学习率调度参数
    alpha_max = args.learning_rate
    alpha_min = args.alpha_min
    T_w = args.warmup_steps
    T_c = args.max_steps
    
    step = 0
    best_val_loss = float('inf')
    learning_curve = []
    
    logger.info(f"开始 batch_size {batch_size_value} 的训练循环...")
    
    while step < args.max_steps:
        # 获取批次数据 (使用传入的 batch_size)
        x, y = transformer.data_loading(train_data, batch_size_value, args.context_length, args.device)
        
        # 计算当前学习率
        current_lr = transformer.learning_rate_schedule(
            step, alpha_max, alpha_min, T_w, T_c
        )
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 前向传播
        logits = model(x)
        
        # 损失计算
        loss = compute_loss(logits, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        transformer.gradient_clipping(model.parameters(), args.gradient_clip)
        
        # 优化器步骤
        optimizer.step()
        
        step += 1

        if step % 500 == 0:
            logger.info(f"[Batch Size {batch_size_value}] Step {step}/{args.max_steps}: train_loss={loss.item():.4f}, lr={current_lr:.6f}")
        
        # 定期保存检查点
        if step % args.save_interval == 0:
            checkpoint_name = f"checkpoint_bs_{batch_size_value}_step_{step}.pt"
            transformer.save_checkpoint(model, optimizer, step, checkpoint_name)
        
        # 定期验证和日志
        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, batch_size_value, args.context_length, args.device)
            curve_data = log_metrics(step, loss.item(), val_loss, current_lr, batch_size_value)
            learning_curve.append(curve_data)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_name = f"best_checkpoint_bs_{batch_size_value}.pt"
                transformer.save_checkpoint(model, optimizer, step, best_checkpoint_name)
    
    logger.info(f"batch_size {batch_size_value} 训练完成，最佳验证损失: {best_val_loss:.4f}")
    return best_val_loss, learning_curve

def batch_size_sweep():
    """Batch size 搜索主函数"""
    logger.info("开始解析参数...")
    args = parse_args()
    logger.info("参数解析完成")
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.results_dir, f"batch_size_sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    logger.info("开始加载训练数据...")
    train_data = load_data_memmap(args.train_path)
    logger.info(f"训练数据加载完成，数据形状: {train_data.shape}")
    
    logger.info("开始加载验证数据...")
    val_data = load_data_memmap(args.val_path)
    logger.info(f"验证数据加载完成，数据形状: {val_data.shape}")
    
    # Batch size 搜索结果
    sweep_results = {
        'batch_sizes': args.batch_sizes,
        'results': [],
        'learning_curves': {}
    }
    
    logger.info(f"开始 batch size 搜索，共 {len(args.batch_sizes)} 个配置")
    
    for bs in args.batch_sizes:
        try:
            best_val_loss, learning_curve = train_single_batch_size(bs, args, train_data, val_data)
            
            # 记录结果
            result = {
                'batch_size': bs,
                'best_val_loss': best_val_loss,
                'achieved_target': best_val_loss <= 1.45
            }
            sweep_results['results'].append(result)
            sweep_results['learning_curves'][f"bs_{bs}"] = learning_curve
            
            logger.info(f"batch_size {bs}: 最终验证损失 = {best_val_loss:.4f}, 达到目标 = {best_val_loss <= 1.45}")
            
        except Exception as e:
            logger.error(f"batch_size {bs} 训练失败: {str(e)}")
            result = {
                'batch_size': bs,
                'best_val_loss': float('inf'),
                'achieved_target': False,
                'error': str(e)
            }
            sweep_results['results'].append(result)
    
    # 保存搜索结果
    results_file = os.path.join(sweep_dir, 'batch_size_sweep_results.json')
    with open(results_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    logger.info(f"batch size 搜索完成，结果已保存到: {results_file}")
    
    # 打印总结
    logger.info("=== Batch Size 搜索总结 ===")
    successful_results = [r for r in sweep_results['results'] if r['best_val_loss'] < float('inf')]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['best_val_loss'])
        logger.info(f"最佳 batch_size: {best_result['batch_size']}")
        logger.info(f"最佳验证损失: {best_result['best_val_loss']:.4f}")
        logger.info(f"达到目标 (≤1.45): {best_result['achieved_target']}")
    
    return sweep_results

if __name__ == "__main__":
    logger.add('./../../Logs/Assignment1.log')
    batch_size_sweep()