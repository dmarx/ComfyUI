import torch
import comfy.model_management
import comfy.samplers
import comfy.conds
import comfy.utils
import math
import numpy as np

def prepare_noise(latent_image, seed, noise_inds=None):
    """
    Creates random noise given a latent image and a seed. Optionally uses `noise_inds` to generate and select specific noise instances.

    Args:
        latent_image (torch.Tensor): The latent image tensor used as a reference for noise generation.
        seed (int): Seed for random number generation to ensure reproducibility.
        noise_inds (list, optional): Indices to select specific noise instances. If None, generates a single noise instance.

    Returns:
        torch.Tensor: Tensor of generated noise.

    Note: Clarification needed on the structure and purpose of `noise_inds`.
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

def prepare_mask(noise_mask, shape, device):
    """
    Ensures the noise mask is of the proper dimensions matching a given shape.

    Args:
        noise_mask (torch.Tensor): Initial noise mask.
        shape (tuple): Target shape to match for the noise mask.
        device (torch.device): The device to which the noise mask should be moved.

    Returns:
        torch.Tensor: Noise mask resized to the specified shape and moved to the designated device.
    """
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = noise_mask.round()
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = comfy.utils.repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask

def get_models_from_cond(cond, model_type):
    """
    Extracts models of a specific type from a list of conditions.

    Args:
        cond (list): List of conditions from which to extract models.
        model_type (str): The type of model to extract.

    Returns:
        list: A list of models of the specified type extracted from the conditions.

    Note: Need more context on the structure of `cond` and the expected model types.
    """
    models = []
    for c in cond:
        if model_type in c:
            models += [c[model_type]]
    return models

def convert_cond(cond):
    """
    Converts a list of conditions into a new format, particularly for use in model conditioning.

    Args:
        cond (list): List of conditions to convert.

    Returns:
        list: List of converted conditions.

    Note: Detailed explanation of the conversion process and the structure of conditions is needed.
    """
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = comfy.conds.CONDCrossAttn(c[0])
        temp["model_conds"] = model_conds
        out.append(temp)
    return out

def get_additional_models(positive, negative, dtype):
    """
    Loads additional models specified in positive and negative conditions.

    Args:
        positive (list): List of positive conditions containing model information.
        negative (list): List of negative conditions containing model information.
        dtype (torch.dtype): The data type for inference memory calculation.

    Returns:
        tuple: A tuple containing a list of additional models and the total inference memory requirement.

    Note: Clarification on the structure of `positive` and `negative` lists and the role of `dtype` is required.
    """
    control_nets = set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control"))

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()

def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    """
    Prepares the model and conditions for sampling.

    Args:
        model: The model to prepare for sampling. More details required.
        noise_shape (tuple): The shape of the noise to be used.
        positive (list): List of positive conditions.
        negative (list): List of negative conditions.
        noise_mask (torch.Tensor, optional): A noise mask tensor.

    Returns:
        tuple: A tuple containing the real model, positive conditions, negative conditions, noise mask, and additional models.

    Note: Detailed explanation of how the model and conditions are prepared is needed.
    """
    device = model.load_device
    positive = convert_cond(positive)
    negative = convert_cond(negative)

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise_shape, device)

    real_model = None
    models, inference_memory = get_additional_models(positive, negative, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + models, model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:])) + inference_memory)
    real_model = model.model

    return real_model, positive, negative, noise_mask, models


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(model, noise.shape, positive, negative, noise_mask)

    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)

    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.cpu()

    cleanup_additional_models(models)
    cleanup_additional_models(set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")))
    return samples

def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None, callback=None, disable_pbar=False, seed=None):
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(model, noise.shape, positive, negative, noise_mask)
    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)
    sigmas = sigmas.to(model.load_device)

    samples = comfy.samplers.sample(real_model, noise, positive_copy, negative_copy, cfg, model.load_device, sampler, sigmas, model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.cpu()
    cleanup_additional_models(models)
    cleanup_additional_models(set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")))
    return samples

