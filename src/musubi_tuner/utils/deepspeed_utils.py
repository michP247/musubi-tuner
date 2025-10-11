import os
import argparse
import torch
from accelerate import DeepSpeedPlugin
import sys

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def add_deepspeed_arguments(parser: argparse.ArgumentParser):
    # DeepSpeed Arguments. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    parser.add_argument("--deepspeed", action="store_true", help="enable deepspeed training")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="Possible options are 0,1,2,3.")
    parser.add_argument(
        "--offload_optimizer_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3.",
    )
    parser.add_argument(
        "--offload_optimizer_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--zero3_init_flag",
        action="store_true",
        help="Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."
        "Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--zero3_save_16bit_model",
        action="store_true",
        help="Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--fp16_master_weights_and_gradients",
        action="store_true",
        help="fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32.",
    )



def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    # Use accelerate's DeepSpeedPlugin to correctly handle the base config
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )

    # Get the config dictionary that the plugin has built
    ds_config = deepspeed_plugin.deepspeed_config

    # Set batch sizes
    ds_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    ds_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )

    # Set mixed precision
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)

    # Ensure the zero_optimization dictionary exists if we are using any stage of ZeRO
    if args.zero_stage > 0 and "zero_optimization" not in ds_config:
        ds_config["zero_optimization"] = {}
    
    # Configure parameter offloading for ZeRO-3
    if args.zero_stage == 3:
        if args.offload_param_device:
            if "offload_param" not in ds_config["zero_optimization"]:
                ds_config["zero_optimization"]["offload_param"] = {}
            ds_config["zero_optimization"]["offload_param"]["device"] = args.offload_param_device
            ds_config["zero_optimization"]["offload_param"]["pin_memory"] = True
            if args.offload_param_device == "nvme":
                ds_config["zero_optimization"]["offload_param"]["nvme_path"] = args.offload_param_nvme_path
                ds_config["zero_optimization"]["offload_param"]["buffer_size"] = 200000000 
        ds_config["log_trace_cache_warnings"] = True

    # Configure optimizer offloading for ZeRO-2 and ZeRO-3
    if args.zero_stage >= 2:
        if args.offload_optimizer_device:
            if "offload_optimizer" not in ds_config["zero_optimization"]:
                ds_config["zero_optimization"]["offload_optimizer"] = {}
            ds_config["zero_optimization"]["offload_optimizer"]["device"] = args.offload_optimizer_device
            ds_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
            if args.offload_optimizer_device == "nvme":
                ds_config["zero_optimization"]["offload_optimizer"]["nvme_path"] = args.offload_optimizer_nvme_path
    
    # Configure AIO if any NVMe offloading is used
    is_optimizer_nvme_offload = args.zero_stage >= 2 and args.offload_optimizer_device == "nvme"
    is_param_nvme_offload = args.zero_stage == 3 and args.offload_param_device == "nvme"

    if is_optimizer_nvme_offload or is_param_nvme_offload:
        ds_config["aio"] = {
            "single_submit": False, "overlap_events": True, "num_threads": 8,
            "queue_depth": 32, "block_size": 1048576, "use_gds": True
        }
        logger.info("[DeepSpeed] NVMe offloading configured.")

    # Configure optimizer
    ds_config["optimizer"] = {
        "type": "Adam",
        "params": {
            "lr": getattr(args, "unet_lr", args.learning_rate),
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0.01,
        }
    }
    #activation checkpointing
    if args.gradient_checkpointing:
        logger.info("[DeepSpeed] Activating DeepSpeed's activation checkpointing (optimized for single GPU).")
        ds_config["activation_checkpointing"] = {
            "partition_activations": False,  # False for single GPU low-VRAM training
            "cpu_checkpointing": True,  # Offload activation checkpoints to CPU RAM
            "contiguous_memory_optimization": True,  # Reduce memory fragmentation
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        }

    if args.mixed_precision.lower() == "fp16":
        ds_config["fp16"]["initial_scale_power"] = 0

    if args.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin
    
# Accelerate library does not support multiple models for deepspeed. So, we need to wrap multiple models into a single model.
def prepare_deepspeed_model(args: argparse.Namespace, **models):
    # remove None from models
    models = {k: v for k, v in models.items() if v is not None}

    class DeepSpeedWrapper(torch.nn.Module):
        def __init__(self, **kw_models) -> None:
            super().__init__()
            self.models = torch.nn.ModuleDict()

            for key, model in kw_models.items():
                if isinstance(model, list):
                    model = torch.nn.ModuleList(model)
                assert isinstance(
                    model, torch.nn.Module
                ), f"model must be an instance of torch.nn.Module, but got {key} is {type(model)}"
                self.models.update(torch.nn.ModuleDict({key: model}))

        def get_models(self):
            return self.models

    ds_model = DeepSpeedWrapper(**models)
    return ds_model

def prepare_deepspeed_args(args: argparse.Namespace):
    if not args.deepspeed:
        logger.info("DeepSpeed is not enabled. Skipping DeepSpeed argument preparation.")
        return

    # To avoid RuntimeError: DataLoader worker exited unexpectedly with exit code 1.
    args.max_data_loader_n_workers = 1