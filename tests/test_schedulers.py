import sys
import os
import pytest
import torch

# Add code directories to path
sd3_path = os.path.join(os.path.dirname(__file__), "..", "code", "text_to_image_sd3")
sys.path.insert(0, os.path.abspath(sd3_path))

from pcm_fm_deterministic_scheduler import PCMFMDeterministicScheduler
from pcm_fm_stochastic_scheduler import PCMFMStochasticScheduler


def test_pcm_fm_deterministic_scheduler_init():
    scheduler = PCMFMDeterministicScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        pcm_timesteps=50,
    )
    assert scheduler.num_train_timesteps == 1000
    assert scheduler.shift == 3.0
    assert len(scheduler.sigmas) == 50
    assert len(scheduler.timesteps) == 50


def test_pcm_fm_deterministic_scheduler_step():
    scheduler = PCMFMDeterministicScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        pcm_timesteps=50,
    )
    scheduler.set_timesteps(num_inference_steps=4)
    assert len(scheduler.timesteps) == 4

    sample = torch.randn((1, 4, 32, 32))
    model_output = torch.randn((1, 4, 32, 32))
    timestep = scheduler.timesteps[0]

    out = scheduler.step(model_output, timestep, sample)
    assert out.prev_sample.shape == sample.shape
    assert not torch.isnan(out.prev_sample).any()


def test_pcm_fm_stochastic_scheduler_init():
    scheduler = PCMFMStochasticScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        pcm_timesteps=50,
    )
    assert scheduler.num_train_timesteps == 1000
    assert len(scheduler.sigmas) == 50


def test_pcm_fm_stochastic_scheduler_step():
    scheduler = PCMFMStochasticScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        pcm_timesteps=50,
    )
    scheduler.set_timesteps(num_inference_steps=4)
    assert len(scheduler.timesteps) == 4

    sample = torch.randn((1, 4, 32, 32))
    model_output = torch.randn((1, 4, 32, 32))
    timestep = scheduler.timesteps[0]

    out = scheduler.step(model_output, timestep, sample)
    assert out.prev_sample.shape == sample.shape
    assert not torch.isnan(out.prev_sample).any()
