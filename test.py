#!/usr/bin/env python3

import datetime as dt
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse


def load_model(checkpoint_path):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to("cpu")
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def count_params(x):
    return f"{sum(p.numel() for p in x.parameters())}"


@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0), dtype=torch.long, device="cpu")[
        None
    ]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device="cpu")
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


@torch.inference_mode()
def synthesise(model, n_timesteps, text, length_scale, temperature, spks=None):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale,
    )
    print("output.shape", list(output.keys()), output["mel"].shape)
    # merge everything to one dict
    output.update({"start_t": start_t, **text_processed})
    return output


@torch.inference_mode()
def to_waveform(mel, vocoder, denoiser):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_24")


@torch.inference_mode()
def main():
    # Download the files from
    # https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases
    MATCHA_CHECKPOINT = "./matcha_ljspeech.ckpt"
    HIFIGAN_CHECKPOINT = "./generator_v1"
    OUTPUT_FOLDER = "synth_output"

    model = load_model(MATCHA_CHECKPOINT)
    print(f"Model loaded! Parameter count: {count_params(model)}")

    vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
    print(f"Vocoder model loaded! Parameter count: {count_params(vocoder)}")
    denoiser = Denoiser(vocoder, mode="zeros")

    texts = [
        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent."
    ]

    # Number of ODE Solver steps
    n_timesteps = 2

    # Changes to the speaking rate
    length_scale = 1.0

    # Sampling temperature
    temperature = 0.667

    outputs, rtfs = [], []
    rtfs_w = []
    for i, text in enumerate(tqdm(texts)):
        output = synthesise(
            model=model,
            n_timesteps=n_timesteps,
            text=text,
            length_scale=length_scale,
            temperature=temperature,
        )  # , torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output["start_t"]).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])

        # Pretty print
        print(f"{'*' * 53}")
        print(f"Input text - {i}")
        print(f"{'-' * 53}")
        print(output["x_orig"])
        print(f"{'*' * 53}")
        print(f"Phonetised text - {i}")
        print(f"{'-' * 53}")
        print(output["x_phones"])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        rtfs.append(output["rtf"])
        rtfs_w.append(rtf_w)

        # Save the generated waveform
        save_to_folder(i, output, OUTPUT_FOLDER)

    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")


if __name__ == "__main__":
    main()
