"""Native Ref2VA rollout capture and differentiable video replay."""

import functools
import inspect
import math
from pathlib import Path

CONVENTION = "H3_dataward_to_ART_noiseward_v2"


def canonical_velocity(native_velocity):
    return -native_velocity.float()


def generated_video_times(kwargs, condition_rows):
    # Time indices refer to the PACKED text/video/audio sequence, not directly
    # to hidden_states' video-only rows. Use the native layout's video map.
    rows = kwargs["video_indices"][
        condition_rows : kwargs["hidden_states"].shape[1]
    ].long()
    return kwargs["timestep"][kwargs["timestep_indices"][rows].long()]


def canonical_schedule(scheduler, index, sample_dtype):
    import torch

    sigma_grid = scheduler.sigmas[index]
    sigma_next = scheduler.sigmas[index + 1]
    native_t = scheduler.timesteps[index]
    noise_sigma = float(sigma_grid)
    if not (0 < noise_sigma <= 1 and 0 <= float(sigma_next) < noise_sigma):
        raise ValueError(
            "Canonical ART requires strictly decreasing positive noise sigma"
        )
    # Match native step EXACTLY: x0 uses 1-t rounded in sample dtype, while
    # Euler's ratio uses the separately stored sigma grid in compute dtype.
    compute_dtype = (
        torch.float32
        if sample_dtype in (torch.float16, torch.bfloat16)
        else sample_dtype
    )
    sigma_from_native_t = 1 - native_t.to(dtype=sample_dtype)
    ratio = sigma_next.to(dtype=compute_dtype) / sigma_grid.to(dtype=compute_dtype)
    coefficient = (1 - ratio) * sigma_from_native_t.to(dtype=compute_dtype)
    dt = -float(coefficient)
    if not -1 <= dt < 0:
        raise ValueError("Native-effective canonical ODE coefficient must be negative")
    if abs(float(native_t) - (1 - noise_sigma)) > 1e-6:
        raise ValueError("H3 native t/sigma convention changed")
    return dict(
        sigma=noise_sigma,
        dt=dt,
        native_timestep=float(native_t),
        native_dataward_coefficient=float(coefficient),
        convention=CONVENTION,
    )


def native_replay_pool(
    video_scheduler, audio_scheduler, api_points, replay_count=4, expected_forwards=None
):
    if type(api_points) is not int or api_points not in (10, 50):
        raise ValueError("Unsupported API sampling point count")
    if replay_count != 4:
        raise ValueError("Frozen replay count is four")
    counts = []
    for original, shift in ((video_scheduler, 12), (audio_scheduler, 3)):
        if float(original.config.shift) != shift:
            raise ValueError("Original video/audio scheduler shift changed")
        # Same native constructor/config and set_timesteps as H3's preparation
        # block; no model, RNG, GPU allocation or live scheduler mutation.
        cloned = type(original).from_config(original.config)
        cloned.set_timesteps(api_points, device="cpu")
        count = len(cloned.timesteps)
        if count < replay_count or len(cloned.sigmas) != count + 1:
            raise ValueError("Native schedule cannot support four actual replay slots")
        sigmas = [float(x) for x in cloned.sigmas]
        times = [float(x) for x in cloned.timesteps]
        if any(not math.isfinite(x) for x in sigmas + times):
            raise ValueError("Nonfinite native schedule")
        if sigmas[0] != 1.0 or sigmas[-1] != 0.0:
            raise ValueError("Native noise endpoints changed")
        if any(
            not (0 <= nxt < current <= 1.0) for current, nxt in zip(sigmas, sigmas[1:])
        ):
            raise ValueError("Native sigmas must be strictly decreasing")
        if any(abs(t - (1.0 - sigma)) > 1e-6 for t, sigma in zip(times, sigmas)):
            raise ValueError("Native t/sigma convention changed")
        counts.append(count)
    if counts[0] != counts[1]:
        raise ValueError("Native video/audio schedules differ in forward count")
    if expected_forwards is not None and expected_forwards != counts[0]:
        raise ValueError(
            "Configured forward count disagrees with actual native scheduler"
        )
    return tuple(range(counts[0]))


def tree(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: tree(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree(v, device) for v in value)
    return value


class Backend:
    def __init__(self, weights, device):
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        from pipeline.offload import ComponentOffload

        self.device = torch.device(device)
        self.weights = Path(weights)
        self.pipe = ModularPipeline.from_pretrained(
            str(self.weights),
            workflow="ref2va",
            components_manager=ComponentsManager(),
            local_files_only=True,
        )
        self.pipe.register_to_config(
            canvas_short_edge=544,
            canvas_max_pixels=960 * 544,
            reference_image_short_edge=544,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.load_components(
            names=[
                "text_encoder",
                "tokenizer",
                "processor",
                "vae",
                "audio_vae",
                "transformer_ref",
                "scheduler",
                "audio_scheduler",
            ],
            pretrained_model_name_or_path=str(self.weights),
            torch_dtype={
                "text_encoder": torch.bfloat16,
                "transformer_ref": torch.bfloat16,
                "default": torch.float32,
            },
            device_map={
                "text_encoder": "cpu",
                "transformer_ref": "cpu",
                "vae": "cpu",
                "audio_vae": "cpu",
                "default": None,
            },
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        for module in self.pipe.components.values():
            if isinstance(module, torch.nn.Module):
                module.eval().requires_grad_(False)
        frozen_components = {
            k: m for k, m in self.pipe.components.items() if k != "transformer_ref"
        }
        self.offload = ComponentOffload(frozen_components, device=self.device)
        self.model = self.pipe.transformer_ref
        self.model.to(self.device)
        self.pipe.vae.enable_tiling()
        self.capturing = False
        self.counter = 0
        self.selected = set()
        self.cache = []
        native = self.model.forward
        self.native_forward = native

        @functools.wraps(native)
        def capture_forward(*args, **kwargs):
            slot = self.counter
            snapshot = (
                tree(kwargs, "cpu")
                if self.capturing and slot in self.selected
                else None
            )
            result = native(*args, **kwargs)
            if self.capturing:
                self.counter += 1
                if snapshot is not None:
                    self.cache.append(
                        dict(
                            step=slot,
                            kwargs=snapshot,
                            old_video=result[0].detach().cpu().clone(),
                        )
                    )
            return result

        self.model.forward = capture_forward
        if set(inspect.signature(native).parameters) != set(
            inspect.signature(self.model.forward).parameters
        ):
            raise ValueError(
                "Hook must preserve native ABI for packed layout field filtering"
            )
        self.adapter_added = False

    def add_lora(self, rank=16, alpha=16):
        import torch
        from peft import LoraConfig

        if self.adapter_added or getattr(self.model, "peft_config", None):
            raise ValueError("Fresh LoRA required; do not silently stack adapters")
        # Each of 50 native self-attention blocks, not AdaLN or modality heads.
        targets = [
            f"transformer_blocks.{i}.attn.{suffix}"
            for i in range(50)
            for suffix in ["to_q", "to_k", "to_v", "to_out.0"]
        ]
        modules = dict(self.model.named_modules())
        if any(not isinstance(modules.get(n), torch.nn.Linear) for n in targets):
            raise ValueError("LoRA target module mismatch with original full-size H3")
        self.model.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=targets,
                init_lora_weights="gaussian",
                lora_dropout=0.0,
                bias="none",
                use_dora=False,
                use_rslora=False,
            ),
            adapter_name="default",
        )
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "lora_" not in name:
                    raise ValueError(f"Unexpected trainable backbone parameter: {name}")
                param.data = param.data.float()
                if "lora_B" in name and torch.count_nonzero(param).item():
                    raise ValueError("LoRA output must initialize exactly zero")
        self.adapter_added = True
        self.model.enable_gradient_checkpointing()

    def replay_pool(self, steps, expected_forwards=None):
        return native_replay_pool(
            self.pipe.scheduler,
            self.pipe.audio_scheduler,
            steps,
            expected_forwards=expected_forwards,
        )

    def rollout(self, frames, mask_video, seed, selected=(), steps=10):
        import torch
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            patchify_video_latents,
        )

        pool = self.replay_pool(steps)
        if len(set(selected)) != len(selected) or not set(selected) <= set(pool):
            raise ValueError(
                "Replay selection must be unique actual native forward slots"
            )
        from infer import PROMPT

        refs = [
            MiniMaxH3VideoReference(frames=v, fps=24.0) for v in (frames, mask_video)
        ]
        self.capturing, self.counter, self.selected, self.cache = (
            True,
            0,
            set(selected),
            [],
        )
        self.model.eval()
        try:
            with torch.no_grad():
                result = self.pipe(
                    prompt=PROMPT,
                    references=refs,
                    height=544,
                    width=960,
                    num_frames=124,
                    num_inference_steps=steps,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    output=[
                        "videos",
                        "latents",
                        "audio_latents",
                        "num_condition_video_rows",
                    ],
                )
        finally:
            self.capturing = False
        self.offload.close()
        if len(self.cache) != len(selected):
            raise ValueError(
                f"Rollout missed selected actual forwards {len(self.cache)} != {len(selected)}"
            )
        final = result["latents"]
        if final.ndim != 5:
            raise ValueError("Native endpoint must be unpacked generated latents")
        packed = patchify_video_latents(
            final, tuple(self.model.config.patch_size)
        ).cpu()
        generated_rows = packed.shape[0]
        if self.counter != len(self.pipe.scheduler.timesteps) or self.counter != len(
            pool
        ):
            raise ValueError(
                "Model forward counter differs from actual native video schedule"
            )
        for cache in self.cache:
            # Generated rows are native suffix; first frame of reference is NOT
            # slot0 of this generated video. Never train conditioning rows.
            n = cache["kwargs"]["hidden_states"].shape[1] - generated_rows
            if n < 0:
                raise ValueError("Negative condition offset")
            cache["condition_rows"] = n
            cache.update(
                canonical_schedule(
                    self.pipe.scheduler,
                    cache["step"],
                    cache["kwargs"]["hidden_states"].dtype,
                )
            )
            # Verify the cached VIDEO suffix actually received this native
            # time, not the clean reference time or the separate audio time.
            video_times = generated_video_times(cache["kwargs"], n)
            if video_times.numel() != generated_rows or not torch.allclose(
                video_times,
                torch.full_like(video_times, cache["native_timestep"]),
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError(
                    "Generated video rows do not match native video schedule clock"
                )
        return dict(
            videos=result["videos"],
            endpoint=packed[None],
            cache=self.cache,
            native_forwards=self.counter,
            sampling_api_points=steps,
            seed=seed,
        )

    def replay(self, cache):
        kwargs = tree(cache["kwargs"], self.device)
        self.offload.close()
        return self.model(**kwargs)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]
