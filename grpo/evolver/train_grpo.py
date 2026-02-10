#!/usr/bin/env python3
# GRPO Training Script for AOI Evolver
"""
Training script for Group Relative Policy Optimization (GRPO).

Usage:
    # Basic training
    python train_grpo.py --seed-dir data/gt/gt_c/claude-sonnet-4.5

    # Custom configuration
    python train_grpo.py \\
        --seed-dir data/gt/gt_c/claude-sonnet-4.5 \\
        --policy-model Qwen/Qwen2.5-7B-Instruct \\
        --reward-model anthropic/claude-sonnet-4-20250514 \\
        --group-size 4 \\
        --batch-size 2 \\
        --learning-rate 1e-5 \\
        --num-epochs 3

    # Resume from checkpoint
    python train_grpo.py --resume-from checkpoints/grpo/checkpoint-1000
"""
import os

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolver.grpo_config import GRPOConfig, RewardScoreConfig
from evolver.grpo_trainer import GRPOTrainer

# Configure logging
class TrainingMetricsFilter(logging.Filter):
    """Filter to only show training metrics in non-verbose mode."""
    
    # Keywords that indicate training metrics we want to show
    SHOW_KEYWORDS = [
        # Training progress
        '[Epoch', '[Step', '[Batch', 'Batch ',
        'loss=', 'reward=', 'accept=',
        # Metrics
        'Total Loss', 'Policy Loss', 'KL Loss', 'Entropy Loss',
        'Mean Reward', 'Acceptance Rate', 'Learning Rate',
        'Group Mean Reward', 'Candidate',
        # Status
        'Epoch', 'Summary', 'Training Complete', 'Best Reward',
        'Checkpoint saved', 'trainable params',
        'LoRA', 'Gradient checkpointing',
        'Starting GRPO', 'Setting up', 'loaded successfully',
        # Formatting
        '═', '─', 'Configuration',
        # Errors/Warnings we want to see
        'Failed to parse', 'Skipping candidate', 'WARNING',
        # Seed info
        'Seed ', 'format=', 'syntax=', 'logic=', 'overall=',
    ]
    
    # Loggers to completely suppress in non-verbose mode
    SUPPRESS_LOGGERS = [
        'httpx', 'httpcore', 'urllib3', 'asyncio',
        'transformers', 'accelerate', 'torch',
    ]
    
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
    
    def filter(self, record: logging.LogRecord) -> bool:
        if self.verbose:
            return True
        
        # Suppress noisy third-party loggers
        for logger_name in self.SUPPRESS_LOGGERS:
            if record.name.startswith(logger_name):
                return False
        
        # Always show errors and warnings
        if record.levelno >= logging.WARNING:
            return True
        
        # Check if message contains training metrics keywords
        message = record.getMessage()
        for keyword in self.SHOW_KEYWORDS:
            if keyword in message:
                return True
        
        return False


def setup_logging(log_dir: str, verbose: bool = False):
    """Setup logging configuration."""
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create console handler with filter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.addFilter(TrainingMetricsFilter(verbose=verbose))
    
    if verbose:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        # Cleaner format for non-verbose mode
        console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler always logs everything
    file_handler = logging.FileHandler(
        Path(log_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers at the logger level
    if not verbose:
        for logger_name in ['httpx', 'httpcore', 'urllib3', 'transformers.tokenization_utils']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train AOI Evolver using GRPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_grpo.py --seed-dir data/gt/gt_c/claude-sonnet-4.5

    # Training with custom settings
    python train_grpo.py \\
        --seed-dir data/gt/gt_c/claude-sonnet-4.5 \\
        --policy-model Qwen/Qwen2.5-7B-Instruct \\
        --group-size 4 \\
        --num-epochs 3

    # Resume training
    python train_grpo.py --resume-from checkpoints/grpo/checkpoint-500
        """
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--seed-dir', '-d',
        type=str,
        default='./data/gt/gt_c/claude-sonnet-4.5',
        help='Directory containing seed JSON files'
    )
    data_group.add_argument(
        '--max-seeds',
        type=int,
        default=None,
        help='Maximum number of seeds to use (default: all)'
    )
    data_group.add_argument(
        '--output-dir',
        type=str,
        default='./data/gt/grpo_output',
        help='Output directory for generated scenarios'
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--policy-model', '-m',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Policy model name or path'
    )
    model_group.add_argument(
        '--policy-model-path',
        type=str,
        default=None,
        help='Local path to policy model (overrides --policy-model)'
    )
    model_group.add_argument(
        '--reward-model',
        type=str,
        default='anthropic/claude-sonnet-4-20250514',
        help='Reward model name'
    )
    model_group.add_argument(
        '--reward-api-base',
        type=str,
        default='https://openrouter.ai/api/v1',
        help='Reward model API base URL'
    )
    model_group.add_argument(
        '--reward-api-key',
        type=str,
        default=None,
        help='Reward model API key (default: from env)'
    )
    
    # GRPO hyperparameters
    grpo_group = parser.add_argument_group('GRPO')
    grpo_group.add_argument(
        '--group-size', '-g',
        type=int,
        default=4,
        help='Number of candidates per seed (default: 4)'
    )
    grpo_group.add_argument(
        '--kl-coef',
        type=float,
        default=0.1,
        help='KL divergence coefficient (default: 0.1)'
    )
    grpo_group.add_argument(
        '--clip-ratio',
        type=float,
        default=0.2,
        help='Policy gradient clip ratio (default: 0.2)'
    )
    grpo_group.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='Entropy bonus coefficient (default: 0.01)'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument(
        '--batch-size', '-b',
        type=int,
        default=2,
        help='Batch size (seeds per batch) (default: 2)'
    )
    train_group.add_argument(
        '--gradient-accumulation', '-ga',
        type=int,
        default=4,
        help='Gradient accumulation steps (default: 4)'
    )
    train_group.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=1e-5,
        help='Learning rate (default: 1e-5)'
    )
    train_group.add_argument(
        '--num-epochs', '-e',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    train_group.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum training steps (overrides --num-epochs)'
    )
    train_group.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='Number of warmup steps (default: 100)'
    )
    train_group.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping (default: 1.0)'
    )
    
    # LoRA arguments
    lora_group = parser.add_argument_group('LoRA')
    lora_group.add_argument(
        '--use-lora',
        action='store_true',
        default=True,
        help='Use LoRA for efficient fine-tuning (default: True)'
    )
    lora_group.add_argument(
        '--no-lora',
        action='store_true',
        help='Disable LoRA (full fine-tuning)'
    )
    lora_group.add_argument(
        '--lora-rank', '-r',
        type=int,
        default=64,
        help='LoRA rank (default: 64)'
    )
    lora_group.add_argument(
        '--lora-alpha',
        type=int,
        default=128,
        help='LoRA alpha (default: 128)'
    )
    lora_group.add_argument(
        '--lora-dropout',
        type=float,
        default=0.05,
        help='LoRA dropout (default: 0.05)'
    )
    
    # Generation arguments
    gen_group = parser.add_argument_group('Generation')
    gen_group.add_argument(
        '--gen-temperature',
        type=float,
        default=0.8,
        help='Generation temperature (default: 0.8)'
    )
    gen_group.add_argument(
        '--gen-top-p',
        type=float,
        default=0.95,
        help='Generation top-p (default: 0.95)'
    )
    gen_group.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum new tokens to generate (default: 2048)'
    )
    
    # Logging and checkpointing
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument(
        '--log-dir',
        type=str,
        default='./logs/grpo',
        help='Log directory'
    )
    log_group.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/grpo',
        help='Checkpoint directory'
    )
    log_group.add_argument(
        '--logging-steps',
        type=int,
        default=10,
        help='Log every N steps (default: 10)'
    )
    log_group.add_argument(
        '--save-steps',
        type=int,
        default=500,
        help='Save checkpoint every N steps (default: 500)'
    )
    log_group.add_argument(
        '--save-total-limit',
        type=int,
        default=3,
        help='Maximum checkpoints to keep (default: 3)'
    )
    log_group.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    log_group.add_argument(
        '--wandb-project',
        type=str,
        default='aoi-grpo',
        help='Wandb project name'
    )
    
    # Hardware arguments
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    hw_group.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision'
    )
    hw_group.add_argument(
        '--bf16',
        action='store_true',
        default=True,
        help='Use BF16 precision (default: True)'
    )
    hw_group.add_argument(
        '--no-bf16',
        action='store_true',
        help='Disable BF16'
    )
    
    # Other arguments
    other_group = parser.add_argument_group('Other')
    other_group.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    other_group.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from checkpoint path'
    )
    other_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    other_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run (setup only, no training)'
    )
    
    return parser.parse_args()


def create_config(args) -> GRPOConfig:
    """Create GRPOConfig from command line arguments."""
    config = GRPOConfig(
        # Model
        policy_model_name=args.policy_model,
        policy_model_path=args.policy_model_path,
        reward_model_api_base=args.reward_api_base,
        reward_model_api_key=args.reward_api_key or os.getenv("OPENROUTER_API_KEY", ""),
        reward_model_name=args.reward_model,
        
        # GRPO
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        
        # Generation
        generation_temperature=args.gen_temperature,
        generation_top_p=args.gen_top_p,
        max_new_tokens=args.max_new_tokens,
        
        # Training
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        
        # LoRA
        use_lora=args.use_lora and not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        
        # Data
        seed_data_dir=args.seed_dir,
        output_dir=args.output_dir,
        max_seeds=args.max_seeds,
        
        # Logging
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        
        # Hardware
        device=args.device,
        fp16=args.fp16,
        bf16=args.bf16 and not args.no_bf16,
        
        # Other
        seed=args.seed,
        resume_from_checkpoint=args.resume_from,
    )
    
    return config


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.verbose)
    
    # Print banner
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    AOI GRPO Training                          ║
║         Group Relative Policy Optimization                    ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Create configuration
    config = create_config(args)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Policy Model: {config.policy_model_name}")
    logger.info(f"  Reward Model: {config.reward_model_name}")
    logger.info(f"  Group Size: {config.group_size}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Num Epochs: {config.num_epochs}")
    logger.info(f"  Use LoRA: {config.use_lora}")
    if config.use_lora:
        logger.info(f"    LoRA Rank: {config.lora_rank}")
        logger.info(f"    LoRA Alpha: {config.lora_alpha}")
    logger.info(f"  Seed Data: {config.seed_data_dir}")
    logger.info(f"  Output Dir: {config.output_dir}")
    logger.info(f"  Checkpoint Dir: {config.checkpoint_dir}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Precision: {'BF16' if config.bf16 else 'FP16' if config.fp16 else 'FP32'}")
    
    # Calculate effective batch size
    effective_batch = config.batch_size * config.gradient_accumulation_steps * config.group_size
    logger.info(f"  Effective Batch Size: {effective_batch}")
    
    # Check API key
    if not config.reward_model_api_key:
        logger.error("No reward model API key provided!")
        logger.error("Set OPENROUTER_API_KEY environment variable or use --reward-api-key")
        sys.exit(1)
    
    # Initialize W&B if requested
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                config=vars(config),
                name=f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            logger.info("Weights & Biases initialized")
        except ImportError:
            logger.warning("wandb not installed, disabling W&B logging")
            config.use_wandb = False
    
    # Create trainer
    reward_config = RewardScoreConfig()
    trainer = GRPOTrainer(config, reward_config)
    
    try:
        # Setup
        logger.info("\nSetting up trainer...")
        trainer.setup()
        
        if args.dry_run:
            logger.info("Dry run complete. Exiting.")
            return
        
        # Train
        logger.info("\nStarting training...")
        history = await trainer.train()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Total Steps: {trainer.global_step}")
        logger.info(f"Best Reward: {trainer.best_reward:.2f}")
        logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")
        logger.info(f"Logs saved to: {config.log_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        await trainer.cleanup()
        
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())

