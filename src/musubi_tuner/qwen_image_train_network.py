import argparse
import gc
from typing import Optional
from PIL import Image


from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_QWEN_IMAGE, ARCHITECTURE_QWEN_IMAGE_FULL
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl, qwen_image_model, qwen_image_utils
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
import logging

from musubi_tuner.utils import image_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QwenImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.is_deepspeed = False

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0  # not used

    def calculate_and_show_nfn_scores(self, args):
        self.handle_model_specific_args(args)

        from musubi_tuner.hv_train_network import prepare_accelerator, collator_class
        from musubi_tuner.dataset import config_utils
        from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
        from musubi_tuner.utils import model_utils
        from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
        from multiprocessing import Value
        import os

        # Set train_batch_size for DeepSpeed plugin (default to 1 for NFN calculation)
        if not hasattr(args, 'train_batch_size'):
            args.train_batch_size = 1

        accelerator = prepare_accelerator(args)

        # dataset
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # vae
        vae_dtype = torch.bfloat16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

        # transformer
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.sage_attn:
            attn_mode = "sageattn"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.flash3:
            attn_mode = "flash3"
        else:
            raise ValueError(
                f"either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified"
            )

        dit_weight_dtype = (None if args.fp8_scaled else torch.float8_e4m3fn) if args.fp8_base else self.dit_dtype

        # For NFN calculation with DeepSpeed, ignore blocks_to_swap and let DeepSpeed handle memory
        using_deepspeed = hasattr(args, 'deepspeed') and args.deepspeed

        if using_deepspeed:
            logger.info("DeepSpeed enabled for NFN calculation. Block swapping will be disabled.")
            blocks_to_swap = 0
            loading_device = accelerator.device
        else:
            blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
            loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        self.blocks_to_swap = blocks_to_swap

        transformer = self.load_transformer(
            accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype
        )

        # Convert to bfloat16 for NFN calculation if loaded in FP8
        # FP8 doesn't support all operations needed for gradient calculation
        # Do this BEFORE device movement to avoid dtype/device mismatch
        target_dtype = torch.bfloat16
        if dit_weight_dtype is not None and dit_weight_dtype == torch.float8_e4m3fn:
            logger.info(f"Converting model from {dit_weight_dtype} to {target_dtype} for NFN calculation")
            transformer.to(dtype=target_dtype)
        elif transformer.dtype != target_dtype:
            logger.info(f"Converting model from {transformer.dtype} to {target_dtype} for NFN calculation")
            transformer.to(dtype=target_dtype)

        transformer.requires_grad_(False)
        transformer.eval()

        # For DeepSpeed Stage 3, we need a parameter to optimize and use accelerator.prepare
        if using_deepspeed:
            from accelerate.utils import DummyOptim

            # Deepspeed needs a parameter to optimize, so we set requires_grad=True for a small parameter
            trainable_param = None
            for name, param in transformer.named_parameters():
                if "bias" in name:
                    trainable_param = param
                    break

            if trainable_param is None:
                trainable_param = next(transformer.parameters())

            trainable_param.requires_grad = True

            dummy_optimizer = DummyOptim([trainable_param])
            transformer, dummy_optimizer = accelerator.prepare(transformer, dummy_optimizer)
        elif blocks_to_swap > 0:
            logger.info(f"enable swap {blocks_to_swap} blocks for NFN calculation")
            transformer.enable_block_swap(blocks_to_swap, accelerator.device, supports_backward=True)
            transformer.move_to_device_except_swap_blocks(accelerator.device)
            transformer.prepare_block_swap_before_forward()
        else:
            # No block swapping, no DeepSpeed, move entire model to device
            transformer.to(device=accelerator.device)

        # noise scheduler
        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

        # Now call the parent's method
        super().calculate_and_show_nfn_scores(args, accelerator, transformer, vae, train_dataloader, noise_scheduler)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load Qwen2.5-VL
        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        tokenizer, text_encoder = qwen_image_utils.load_qwen2_5_vl(args.text_encoder, vl_dtype, device, disable_mmap=True)
        is_edit = args.edit
        vl_processor = qwen_image_utils.load_vl_processor() if is_edit else None

        # Encode with VLM
        logger.info(f"Encoding with VLM")

        sample_prompts_te_outputs = {}  # prompt -> embed or (prompt, control_image_path) -> embed
        with torch.amp.autocast(device_type=device.type, dtype=vl_dtype), torch.no_grad():
            for prompt_dict in prompts:
                if is_edit:
                    # Load control image
                    assert (
                        "control_image_path" in prompt_dict and len(prompt_dict["control_image_path"]) > 0
                    ), f"control_image_path not found in sample prompt"
                    control_image_path = prompt_dict["control_image_path"][0]  # only use the first control image
                    control_image_tensor, control_image_np, _ = qwen_image_utils.preprocess_control_image(control_image_path, True)

                    prompt_dict["control_image_tensor"] = control_image_tensor
                else:
                    control_image_path, control_image_tensor, control_image_np = None, None, None

                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = " "

                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", " ")]:
                    embed_key = p if not is_edit else (p, control_image_path)
                    if p is None or embed_key in sample_prompts_te_outputs:
                        continue

                    # encode prompt with image if available
                    logger.info(f"cache Text Encoder outputs for prompt: {p} with image: {control_image_path}")
                    if not is_edit:
                        embed, mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, p)
                    else:
                        embed, mask = qwen_image_utils.get_qwen_prompt_embeds_with_image(
                            vl_processor, text_encoder, p, control_image_np
                        )
                    txt_len = mask.to(dtype=torch.bool).sum().item()  # length of the text in the batch
                    embed = embed[:, :txt_len]
                    sample_prompts_te_outputs[embed_key] = embed

        del tokenizer, text_encoder
        gc.collect()
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            control_image_path = None if not is_edit else prompt_dict_copy["control_image_path"][0]

            p = prompt_dict.get("prompt", "")
            embed_key = p if not is_edit else (p, control_image_path)
            prompt_dict_copy["vl_embed"] = sample_prompts_te_outputs[embed_key]

            p = prompt_dict.get("negative_prompt", "")
            embed_key = p if not is_edit else (p, control_image_path)
            prompt_dict_copy["negative_vl_embed"] = sample_prompts_te_outputs[embed_key]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """architecture dependent inference"""
        model: qwen_image_model.QwenImageTransformer2DModel = transformer
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae
        is_edit = args.edit

        device = accelerator.device

        if cfg_scale is None:
            cfg_scale = 4.0

        # Get embeddings
        vl_embed = sample_parameter["vl_embed"].to(device=device, dtype=torch.bfloat16)
        txt_seq_lens = [vl_embed.shape[1]]
        negative_vl_embed = sample_parameter["negative_vl_embed"].to(device=device, dtype=torch.bfloat16)
        negative_txt_seq_lens = [negative_vl_embed.shape[1]]

        # 4. Prepare latent variables
        num_channels_latents = model.in_channels // 4
        # latents is packed
        latents = qwen_image_utils.prepare_latents(1, num_channels_latents, height, width, torch.bfloat16, device, generator)
        img_shapes = [(1, height // qwen_image_utils.VAE_SCALE_FACTOR // 2, width // qwen_image_utils.VAE_SCALE_FACTOR // 2)]

        if is_edit:
            # 4.1 Prepare control latents
            logger.info(f"Preparing control latents from control image")
            control_image_tensor = sample_parameter.get("control_image_tensor")
            vae.to(device)
            vae.eval()

            with torch.no_grad():
                control_latent = vae.encode_pixels_to_latents(control_image_tensor.to(device, vae.dtype))
            control_latent = control_latent.to(torch.bfloat16).to("cpu")

            vae.to("cpu")
            clean_memory_on_device(device)

            img_shapes = [[img_shapes[0], (1, control_latent.shape[-2] // 2, control_latent.shape[-1] // 2)]]
            control_latent = qwen_image_utils.pack_latents(control_latent)
            control_latent = control_latent.to(device=device, dtype=torch.bfloat16)
        else:
            control_latent = None

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
        image_seq_len = latents.shape[1]

        mu = qwen_image_utils.calculate_shift_qwen_image(image_seq_len)
        scheduler = qwen_image_utils.get_scheduler(discrete_flow_shift)
        # mu is kwarg for FlowMatchingDiscreteScheduler
        timesteps, n = qwen_image_utils.retrieve_timesteps(scheduler, sample_steps, device, sigmas=sigmas, mu=mu)
        assert n == sample_steps, f"Expected steps={sample_steps}, got {n} from scheduler."

        num_warmup_steps = 0  # because FlowMatchingDiscreteScheduler.order is 1, we don't need warmup steps

        # handle guidance
        guidance = None  # guidance_embeds is false for Qwen-Image

        # 6. Denoising loop
        do_cfg = do_classifier_free_guidance and cfg_scale > 1.0
        scheduler.set_begin_index(0)
        # with progress_bar(total=sample_steps) as pbar:
        with tqdm(total=sample_steps, desc="Denoising steps") as pbar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents
                if is_edit:
                    latent_model_input = torch.cat([latents, control_latent], dim=1)

                with torch.no_grad():
                    noise_pred = model(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=None,
                        encoder_hidden_states=vl_embed,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                    )
                    if is_edit:
                        noise_pred = noise_pred[:, :image_seq_len]

                if do_cfg:
                    with torch.no_grad():
                        neg_noise_pred = model(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=None,
                            encoder_hidden_states=negative_vl_embed,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                        )
                    if is_edit:
                        neg_noise_pred = neg_noise_pred[:, :image_seq_len]
                    comb_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    pbar.update()

        latents = qwen_image_utils.unpack_latents(latents, height, width)

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        logger.info(f"Decoding video from latents: {latents.shape}")
        with torch.no_grad():
            pixels = vae.decode_to_pixels(latents.to(device))  # decode to pixels, 0-1
        del latents

        logger.info(f"Decoding complete")
        pixels = pixels.to(torch.float32).cpu()

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2)  # add a dummy dimension for video frames, B C H W -> B C 1 H W
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading VAE model from {vae_path}")
        vae = qwen_image_utils.load_vae(args.vae, device="cpu", disable_mmap=True)
        vae.eval()
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        try:
            from safetensors.torch import load_file
            from tqdm import tqdm
        except ImportError:
            pass

        is_deepspeed_stage3 = (
            self.is_deepspeed
            and hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        if is_deepspeed_stage3:
            import deepspeed

            logger.info("DeepSpeed Stage 3 detected. Creating hollow QwenImage model and injecting weights chunk-by-chunk.")
            ds_config = accelerator.state.deepspeed_plugin.deepspeed_config

            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = qwen_image_model.load_qwen_image_model(
                    accelerator.device,
                    dit_path,
                    attn_mode,
                    split_attn,
                    "cpu",  # loading_device
                    dit_weight_dtype,
                    args.fp8_scaled,
                    load_weights=False,  # create a hollow model
                )
            logger.info("Hollow QwenImage model created.")

            logger.info(f"Loading original weights from: {dit_path} into CPU RAM...")
            original_sd = load_file(dit_path, device="cpu")

            # remove "model.diffusion_model." prefix
            for key in list(original_sd.keys()):
                if key.startswith("model.diffusion_model."):
                    original_sd[key[22:]] = original_sd.pop(key)

            logger.info("Injecting original weights into the sharded model...")
            all_params = list(model.parameters())
            chunk_size = 8
            for i in tqdm(
                range(0, len(all_params), chunk_size),
                desc="Injecting weight chunks",
                disable=not accelerator.is_main_process,
            ):
                chunk = all_params[i : i + chunk_size]
                with deepspeed.zero.GatheredParameters(chunk, modifier_rank=0):
                    if accelerator.is_main_process:
                        for j, param in enumerate(chunk):
                            # Find the parameter's original name
                            param_name = [name for name, p in model.named_parameters() if p is all_params[i + j]][0]
                            if param_name in original_sd:
                                param.data.copy_(original_sd[param_name].to(param.device, param.dtype))

            logger.info("Weight injection complete.")
        else:
            model = qwen_image_model.load_qwen_image_model(
                accelerator.device, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype, args.fp8_scaled
            )
        return model

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        model: qwen_image_model.QwenImageTransformer2DModel = transformer
        is_edit = args.edit

        bsize = latents.shape[0]
        latents = batch["latents"]  # B, C, 1, H, W

        # pack latents
        lat_h = latents.shape[3]
        lat_w = latents.shape[4]
        # print(noisy_model_input.shape, bsize, latents.shape[1], lat_h, lat_w)
        noisy_model_input = qwen_image_utils.pack_latents(noisy_model_input)
        img_seq_len = noisy_model_input.shape[1]

        # control
        if is_edit:
            latents_control = batch["latents_control"]  # B, C, 1, H, W
            latents_control_shape = latents_control.shape
            latents_control = qwen_image_utils.pack_latents(latents_control)

            noisy_model_input = torch.cat([noisy_model_input, latents_control], dim=1)  # B, C*2, 1, H, W
        else:
            latents_control, latents_control_shape = None, None

        # context
        vl_embed = batch["vl_embed"]  # list of (L, D)
        txt_seq_lens = [x.shape[0] for x in vl_embed]

        max_len = max(txt_seq_lens)
        vl_embed = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
        vl_embed = torch.stack(vl_embed, dim=0)  # B, L, D

        # if not split_attn, we need to make attention mask
        if not args.split_attn and bsize > 1:
            vl_mask = torch.zeros(bsize, max_len, dtype=torch.bool, device=vl_embed[0].device)
            for i, x in enumerate(txt_seq_lens):
                vl_mask[i, :x] = True
        else:
            vl_mask = None  # if split_attn, vl_mask is not used
        # print(f"vl_embed shape: {vl_embed.shape}, vl_mask shape: {vl_mask.shape if vl_mask is not None else None}")

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            vl_embed.requires_grad_(True)

        # call DiT
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        vl_embed = vl_embed.to(device=accelerator.device, dtype=network_dtype)
        if vl_mask is not None:
            vl_mask = vl_mask.to(device=accelerator.device)  # bool

        img_shapes = [(1, lat_h // 2, lat_w // 2)]
        if is_edit:
            img_shapes = [[img_shapes[0], (1, latents_control_shape[-2] // 2, latents_control_shape[-1] // 2)]]

        guidance = None
        timesteps = timesteps / 1000.0
        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                guidance=guidance,
                encoder_hidden_states_mask=vl_mask,
                encoder_hidden_states=vl_embed,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )
            if is_edit:
                model_pred = model_pred[:, :img_seq_len]

        # unpack latents
        model_pred = qwen_image_utils.unpack_latents(
            model_pred,
            lat_h * qwen_image_utils.VAE_SCALE_FACTOR,
            lat_w * qwen_image_utils.VAE_SCALE_FACTOR,
            qwen_image_utils.VAE_SCALE_FACTOR,
        )

        # flow matching loss
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents

        # print(model_pred.dtype, target.dtype)
        return model_pred, target

    # endregion model specific


def qwen_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Qwen-Image specific parser setup"""
    from musubi_tuner.utils import deepspeed_utils

    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder (Qwen2.5-VL) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument("--edit", action="store_true", help="training for Qwen-Image-Edit")

    # Add DeepSpeed arguments
    deepspeed_utils.add_deepspeed_arguments(parser)

    # NFN arguments
    parser.add_argument("--calculate_nfn_weights", action="store_true", help="calculate NFN weights")
    parser.add_argument(
        "--nfn_min_lr_weight",
        type=float,
        default=0.1,
        help="Minimum learning rate weight for NFN. Default is 0.1.",
    )
    parser.add_argument(
        "--nfn_max_lr_weight",
        type=float,
        default=2.0,
        help="Maximum learning rate weight for NFN. Default is 2.",
    )
    parser.add_argument("--network_block_lr_weights", type=float, nargs="*", help="learning rate weights for each block in the network")

    parser.add_argument(
        "--auto_balance_reg_datasets",
        action="store_true",
        help="auto balance regularization datasets to fill the rest of the deck based on max_train_steps",
    )

    return parser


def main():
    from musubi_tuner.utils import deepspeed_utils

    parser = setup_parser_common()
    parser = qwen_image_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    deepspeed_utils.prepare_deepspeed_args(args)

    args.dit_dtype = "bfloat16"  # DiT dtype is bfloat16
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # make bfloat16 as default for VAE, this should be checked

    trainer = QwenImageNetworkTrainer()
    if hasattr(args, 'deepspeed') and args.deepspeed:
        trainer.is_deepspeed = True

    if args.calculate_nfn_weights:
        trainer.calculate_and_show_nfn_scores(args)
    else:
        trainer.train(args)


if __name__ == "__main__":
    main()
