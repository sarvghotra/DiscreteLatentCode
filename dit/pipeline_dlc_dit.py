from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from diffusers.models import AutoencoderKL, DiTTransformer2DModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, ImagePipelineOutput

from diffusion import create_diffusion


def cfg_wrapper(forward, cfg_scale, in_channels):
    def fn(x, t, y):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, : in_channels], model_out[:, in_channels :]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    return fn


def id_wrapper(forward):
    def fn(x, t, y):
        return forward(x, t, y)

class DLCDiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer->vae"
    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)


    @torch.inference_mode()
    def __call__(
        self,
        dlcs: List[List[int]],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            dlc (List[int]):
                List of Discrete Latent Codes for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import torch

        >>> pipe = DLCDiTPipeline.from_pretrained("lavoies/lavoies/DLC_DiT_L512", trust_remote_code=True)
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.manual_seed(33)
        >>> dlcs = torch.load("dlcs.pt") # Saved DLCs either unconditionally generated or generated from a text-and-DLC model.
        >>> output = pipe(dlcs=dlcs, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # 'dlcs' of a golden retriever
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        diffusion = create_diffusion(str(num_inference_steps))
        batch_size = len(dlcs)
        latent_size = self.transformer.config.input_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        # latents = torch.cat((latents, latents), dim=0)
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        sample_fn = cfg_wrapper(self.transformer, guidance_scale, latent_channels) if guidance_scale > 1 else self.transformer

        dlcs = torch.tensor(dlcs, device=self._execution_device)
        L, V = self.transformer.config.L, self.transformer.config.V
        dlcs_null = torch.tensor([[V] * L] * batch_size, device=self._execution_device)
        dlcs_input = torch.cat([dlcs, dlcs_null], 0) if guidance_scale > 1 else dlcs

        model_kwargs = dict(y=dlcs_input)

        samples = diffusion.p_sample_loop(
            sample_fn,
            latent_model_input.shape,
            latent_model_input,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self._execution_device)

        if guidance_scale > 1:
            latents, _ = samples.chunk(2, dim=0)
        else:
            latents = samples

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        # samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        samples = [Image.fromarray(image) for image in samples]

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        # if output_type == "pil":
        #     samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
