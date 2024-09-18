#!/usr/bin/env bash

wget https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1
wget wget https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt


python3 ./test.py
