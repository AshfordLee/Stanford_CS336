from cs336_basics import transformer
from loguru import logger
import numpy as np
import torch

import argparse

logger.add('./../../Logs/Assignment1.log')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 数据路径
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
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
    
    # 训练超参数 (需要调优)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 需要调优
    parser.add_argument('--weight_decay', type=float, default=0.01)   # 需要调优
    parser.add_argument('--beta1', type=float, default=0.9)           # 需要调优
    parser.add_argument('--beta2', type=float, default=0.999)         # 需要调优
    parser.add_argument('--eps', type=float, default=1e-8)            # 需要调优
    parser.add_argument('--warmup_steps', type=int, default=1000)     # 需要调优
    parser.add_argument('--max_steps', type=int, default=10000)       # 总训练步数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 控制参数
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    
    return parser.parse_args()

def log_metrics(step, train_loss, val_loss, current_lr=None):
    if current_lr is not None:
        logger.info(f"Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")
    else:
        logger.info(f"Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


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

def train():

    logger.info("开始解析参数...")
    args = parse_args()
    logger.info("参数解析完成")

    logger.info("开始创建模型...")
    model = create_model((
        args.d_model, args.num_heads, args.d_ff, args.vocab_size,
        args.context_length, args.num_layers, args.use_rope,
        args.max_seq_len, args.theta
    ))
    logger.info(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters())}")

    logger.info(f"将模型移动到设备: {args.device}")
    model.to(args.device)
    logger.info("模型已移动到设备")

    logger.info("开始创建优化器...")
    optimizer = create_optimizer(model, (
        model.parameters(), args.learning_rate,
        (args.beta1, args.beta2), args.eps, args.weight_decay
    ))
    logger.info("优化器创建完成")

    logger.info("开始加载训练数据...")
    train_data = load_data_memmap(args.train_path)
    logger.info(f"训练数据加载完成，数据形状: {train_data.shape}")

    logger.info("开始加载验证数据...")
    val_data = load_data_memmap(args.val_path)
    logger.info(f"验证数据加载完成，数据形状: {val_data.shape}")

    alpha_max = args.learning_rate  # 最大学习率
    alpha_min = 1e-5  # 最小学习率 (可以作为参数)
    T_w = args.warmup_steps  # warmup步数
    T_c = args.max_steps  # 总步数 (cosine周期结束)

    step = 0
    best_val_loss = float('inf')

    logger.info("Starting training...")
    logger.info(f"训练配置: max_steps={args.max_steps}, batch_size={args.batch_size}, device={args.device}")

    while step < args.max_steps:
        logger.info(f"=== 开始第 {step} 步训练 ===")
        logger.info(f"正在生成批次数据... (步数: {step})")

        batch_start_time = torch.cuda.Event(enable_timing=True) if args.device == 'cuda' else None
        if batch_start_time:
            batch_start_time.record()

        # 获取批次数据
        x, y = transformer.data_loading(train_data, args.batch_size, args.context_length, args.device)
        logger.info(f"批次数据生成完成，数据形状: x={x.shape}, y={y.shape}")

        # 计算当前学习率
        logger.info(f"正在计算学习率... (步数: {step})")
        current_lr = transformer.learning_rate_schedule(
            step, alpha_max, alpha_min, T_w, T_c
        )
        logger.info(f"学习率计算完成: {current_lr:.6f}")

        # 更新优化器学习率
        logger.info("正在更新优化器学习率...")
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        logger.info("优化器学习率更新完成")

        # 前向传播
        logger.info("正在进行前向传播...")
        forward_start = torch.cuda.Event(enable_timing=True) if args.device == 'cuda' else None
        if forward_start:
            forward_start.record()

        logits = model(x)

        if forward_start and torch.cuda.is_available():
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_end.record()
            torch.cuda.synchronize()
            forward_time = forward_start.elapsed_time(forward_end)
            logger.info(f"前向传播完成，耗时: {forward_time:.2f}ms，输出形状: {logits.shape}")

        # 损失计算
        logger.info("正在计算损失...")
        loss = compute_loss(logits, y)
        logger.info(f"损失计算完成: {loss.item():.4f}")

        # 反向传播
        logger.info("正在进行反向传播...")
        optimizer.zero_grad()
        backward_start = torch.cuda.Event(enable_timing=True) if args.device == 'cuda' else None
        if backward_start:
            backward_start.record()

        loss.backward()

        if backward_start and torch.cuda.is_available():
            backward_end = torch.cuda.Event(enable_timing=True)
            backward_end.record()
            torch.cuda.synchronize()
            backward_time = backward_start.elapsed_time(backward_end)
            logger.info(f"反向传播完成，耗时: {backward_time:.2f}ms")

        # 使用你自己的梯度裁剪
        logger.info("正在进行梯度裁剪...")
        transformer.gradient_clipping(model.parameters(), args.gradient_clip)
        logger.info("梯度裁剪完成")

        # 优化器步骤
        logger.info("正在执行优化器步骤...")
        optimizer.step()
        logger.info("优化器步骤完成")

        step += 1

        # 定期保存检查点
        if step % args.save_interval == 0:
            logger.info(f"正在保存检查点... (步数: {step})")
            transformer.save_checkpoint(model, optimizer, step, args.checkpoint_path)
            logger.info("检查点保存完成")

        # 定期验证和日志
        if step % args.eval_interval == 0:
            logger.info(f"正在进行验证... (步数: {step})")
            eval_start = torch.cuda.Event(enable_timing=True) if args.device == 'cuda' else None
            if eval_start:
                eval_start.record()

            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device)

            if eval_start and torch.cuda.is_available():
                eval_end = torch.cuda.Event(enable_timing=True)
                eval_end.record()
                torch.cuda.synchronize()
                eval_time = eval_start.elapsed_time(eval_end)
                logger.info(f"验证完成，耗时: {eval_time:.2f}ms")

            log_metrics(step, loss.item(), val_loss, current_lr)

            # 保存最佳模型
            if val_loss < best_val_loss:
                logger.info(f"发现更好的验证损失: {val_loss:.4f} < {best_val_loss:.4f}，正在保存最佳模型...")
                best_val_loss = val_loss
                transformer.save_checkpoint(model, optimizer, step, f"best_{args.checkpoint_path}")
                logger.info("最佳模型保存完成")

        if step >= args.max_steps:
            logger.info("达到最大步数，准备退出训练循环")
            break

        if batch_start_time and torch.cuda.is_available():
            batch_end_time = torch.cuda.Event(enable_timing=True)
            batch_end_time.record()
            torch.cuda.synchronize()
            batch_time = batch_start_time.elapsed_time(batch_end_time)
            logger.info(f"批次处理完成，总耗时: {batch_time:.2f}ms")

    logger.info("训练循环结束")
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Target validation loss: ≤ 1.45 (achieved: {best_val_loss <= 1.45})")

if __name__ == "__main__":
    logger.add('./../../Logs/Assignment1.log')
    train()