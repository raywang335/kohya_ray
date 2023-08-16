# training with captions

import argparse
import gc
import math
import os
from multiprocessing import Value
import wandb 
from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import sdxl_model_util
import numpy as np
import library.train_util as train_util
import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.sdxl_t2i_adapter import SdxlT2IAdapter
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    pyramid_noise_like,
    apply_noise_offset,
    scale_v_prediction_loss_like_noise_prediction,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from PIL import Image

@torch.no_grad() 
def inference_samples(unet, vae, controller, ori_image_path, text_embeddings, content_embeddings, latents_dict=None):
    # settings
    target_num = 1
    steps = 25
    guidance_scale = 9
    DTYPE = torch.float16
    DEVICE = content_embeddings.device
    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"

    vae.to(DEVICE, dtype=torch.float32)
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
    )
    content_embeddings = content_embeddings[:1, :].to(DEVICE, dtype=DTYPE)

    # text condition TODO: Shape alignment
    cond_encoder_hidden_states1 = text_embeddings['hid1'].to(DEVICE,  dtype=DTYPE)[:1, :, :]
    cond_encoder_hidden_states2 = text_embeddings['hid2'].to(DEVICE,  dtype=DTYPE)[:1, :, :]
    cond_pool2 = text_embeddings['pool'].to(DEVICE,  dtype=DTYPE)[:1, :]

    # ucond: 77*D to tensor
    uc_text_embeddings = np.load(os.path.join("/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/cache/null_token.npy"), allow_pickle=True).item()
    ucond_encoder_hidden_states1 = torch.FloatTensor(uc_text_embeddings['hid1']).unsqueeze(0).repeat(cond_encoder_hidden_states1.shape[0], 1, 1).to(DEVICE,  dtype=DTYPE)
    ucond_encoder_hidden_states2 = torch.FloatTensor(uc_text_embeddings['hid2']).unsqueeze(0).repeat(cond_encoder_hidden_states1.shape[0], 1, 1).to(DEVICE,  dtype=DTYPE)
    ucond_pool2 = torch.FloatTensor(uc_text_embeddings['pool']).repeat(cond_encoder_hidden_states1.shape[0], 1).to(DEVICE, dtype=DTYPE)

    cond_text_embedding = torch.cat([cond_encoder_hidden_states1, cond_encoder_hidden_states2], dim=2)
    ucond_text_embedding = torch.cat([ucond_encoder_hidden_states1, ucond_encoder_hidden_states2], dim=2)

    # other condition
    cond_content_embeddings = torch.cat([cond_pool2, content_embeddings], dim=-1).to(DEVICE, dtype=DTYPE)
    ucond_content_embeddings = torch.cat([ucond_pool2, content_embeddings], dim=-1).to(DEVICE, dtype=DTYPE)

    # combine condition
    to_text_embeddings = torch.cat([ucond_text_embedding, cond_text_embedding])
    to_vector_embeddings = torch.cat([ucond_content_embeddings, cond_content_embeddings])
    latents_shape = (1, 4, controller[0].shape[-2]*2, controller[0].shape[-1]*2)

    if latents_dict is not None:
        latents = latents_dict.get(f"{controller[0].shape[-2]}{controller[0].shape[-1]}", None)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                device="cpu",
                dtype=torch.float32,
            )
            latents_dict[f"{controller[0].shape[-2]}{controller[0].shape[-1]}"] = latents
    else:
        latents = torch.randn(
            latents_shape,
            device="cpu",
            dtype=torch.float32,
        )
    latents = latents.to(DEVICE, dtype=DTYPE)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(steps, DEVICE)
    timesteps = scheduler.timesteps.to(DEVICE)  # .to(DTYPE)
    num_latent_input = 2
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(DEVICE, dtype=DTYPE)
            adapter_features = [i[:1, :, :, :].clone().repeat(2,1,1,1) for i in controller]
            noise_pred = unet(latent_model_input, t, to_text_embeddings, to_vector_embeddings, adapter_features=adapter_features)

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # latents = 1 / 0.18215 * latents
    latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents
    latents = latents.to(torch.float32)
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1) # Bx3xHxW
    # ori_image = np.array(Image.open(ori_image_path))
    # test_images = [(img.cpu().numpy() * 255.).astype(np.uint8) for img in range(target_num)]
    # images = [ori_image].extend(test_images)
    samples = []
    for i in range(target_num):
        ori_image = np.array(Image.open(ori_image_path[i])) # HxWx3 0-255
        sketch_image_path = os.path.join("/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/T2Iadapter_lineart/lineart_coarse_data",os.path.basename(os.path.dirname(ori_image_path[i])), os.path.basename(ori_image_path[i]))
        sketch_image = np.array(Image.open(sketch_image_path).convert('L')) # HxWx3 0-255
        test_image = image[i].permute(1, 2, 0).cpu().numpy() # HxWx3 0-1
        test_image = (test_image * 255.).astype(np.uint8)
        caption_path = os.path.splitext(ori_image_path[i])[0] + ".txt"
        caption_data = open(caption_path, "r").read().strip()
        samples1 = wandb.Image(ori_image, caption=f"Original: {caption_data}")
        samples2 = wandb.Image(test_image, caption=f"Sample: {caption_data}")
        samples3 = wandb.Image(sketch_image, caption=f"Sketch")
        
        samples.append(samples1)
        samples.append(samples2)
        samples.append(samples3)
    return samples, latents_dict



def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)

    assert not args.weighted_captions, "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    cache_latents = args.cache_latents
    text_cache_latents = args.text_encoder_cache
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True, True))
        if args.dataset_config is not None:
            print(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                print(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                print("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                print("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                    "caption_dropout_rate": args.caption_dropout_rate,
                                    "lineart_dir": args.lineart_dir,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        print(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"
    # acceleratorを準備する
    print("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)

    #NOTE: add adapter models
    if args.adapter_resume_path is None:
        sdxl_adapter = SdxlT2IAdapter(sk=True, use_conv=False)
    else:
        assert os.path.exists(args.adapter_resume_path), f"adapter_resume_path {args.adapter_resume_path} does not exist"
        sdxl_adapter = sdxl_train_util.load_xl_model(args, accelerator, 'adapter', weight_dtype)


    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        unet.set_use_memory_efficient_attention(True, False)
        set_diffusers_xformers_flag(vae, True)
        # set_diffusers_xformers_flag(unet, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # # 学習を準備する
    # if cache_latents:
    #     vae.to(accelerator.device, dtype=vae_dtype)
    #     vae.requires_grad_(False)
    #     vae.eval()
    #     # with torch.no_grad():
    #     #     train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
    #     vae.to("cpu")
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     gc.collect()
    #     accelerator.wait_for_everyone()
    vae.to("cpu", dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.eval()

    if text_cache_latents:
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()
        text_encoder1.to('cpu')
        text_encoder2.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        accelerator.wait_for_everyone()


    # 学習を準備する：モデルを適切な状態にする
    training_models = []
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print(f'enable gradient checkpointing for U-Net')
    
    #NOTE: training adapters
    training_models.append(sdxl_adapter)

    for m in training_models:
        m.requires_grad_(True)
    params = []
    for m in training_models:
        params.extend(m.parameters())
    params_to_optimize = params

    # calculate number of trainable parameters
    n_params = 0
    for p in params:
        n_params += p.numel()
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collater,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        sdxl_adapter.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        sdxl_adapter.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    if args.train_text_encoder:
        unet, text_encoder1, text_encoder2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder1, text_encoder2, optimizer, train_dataloader, lr_scheduler
        )

        # transform DDP after prepare
        text_encoder1, text_encoder2, unet = train_util.transform_models_if_DDP([text_encoder1, text_encoder2, unet])
    else:
        sdxl_adapter, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(sdxl_adapter, unet, optimizer, train_dataloader, lr_scheduler)
        sdxl_adapter, unet = train_util.transform_models_if_DDP([sdxl_adapter, unet])
        # text_encoder1.to(weight_dtype)
        # text_encoder2.to(weight_dtype)

    # # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    # if args.cache_text_encoder_outputs:
    #     # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
    #     text_encoder1.to("cpu", dtype=torch.float32)
    #     text_encoder2.to("cpu", dtype=torch.float32)
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    # else:
    #     # make sure Text Encoders are on GPU
    #     text_encoder1.to(accelerator.device)
    #     text_encoder2.to(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers("xl-adapter" if args.log_tracker_name is None else args.log_tracker_name)

    controller = None
    infer_text_embedding = None
    infer_content_embeddings = None
    ori_img_name = None
    fixed_crop_emb = None
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            # with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
            if True:
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                encoder_hidden_states1 = batch['hid1']
                encoder_hidden_states2 = batch['hid2']
                pool2 = batch['pool']
                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)
                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(accelerator.device).to(weight_dtype)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, mode='cubic')

                noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                with accelerator.autocast():
                    # print(batch['adapter_cond'].shape)
                    adapters_features = sdxl_adapter(batch['adapter_cond'])
                    controller = [i.clone().detach() for i in adapters_features]
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding, adapter_features=adapters_features)
                

                infer_text_embedding = {
                    'hid1': encoder_hidden_states1.clone().detach(),
                    'hid2': encoder_hidden_states2.clone().detach(),
                    'pool': pool2.clone().detach(),
                }
                infer_content_embeddings = embs.clone().detach()
                ori_img_name = batch['image_keys']
                
                target = noise

                if args.min_snr_gamma:
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    [tokenizer1, tokenizer2],
                    [text_encoder1, text_encoder2],
                    unet,
                )
                

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_adapter_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(sdxl_adapter),
                            ckpt_info,
                        )
                # 指定ステップごとにモデルをlog image

                if args.log_image_every_n_steps is not None and global_step % args.log_image_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            samples, _ = inference_samples(unet, vae, controller, ori_img_name, infer_text_embedding, infer_content_embeddings)
                            wandb.log({"samples": samples})

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if args.logging_dir is not None:
                logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                if (
                    args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy"
                ):  # tracking d*lr value
                    logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                    )
                accelerator.log(logs, step=global_step)

            # TODO moving averageにする
            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(train_dataloader)}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_adapter_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(sdxl_adapter),
                    ckpt_info,
                )

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2],
            unet,
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.save_state:  # and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        print("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument("--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する")
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--text_encoder_cache",
        action="store_true"
    )
    parser.add_argument(
        "--phrase",
        default=None
    )
    parser.add_argument(
        "--lineart_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--log_image_every_n_steps",
        type=int,
        default=2
    )
    parser.add_argument(
        "--adapter_resume_path",
        type=str,
        default=None,
    )


    
    

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
