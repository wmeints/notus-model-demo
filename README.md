# Notus 7B model demo

This repository contains a set of demo scripts to show how you can use the Notus 7B large language model.
Please follow the instructions below to get started.

## System requirements

Hardware requirements:

- Nvidia GPU with 16GB memory (e.g. RTX4080)
- 32GB RAM

Software requirements:

- Python 3.11
- Linux (WSL2 works too)

## Getting started

Follow these steps to configure your machine:

- `git clone https://github.com/wmeints/notus-model-demo.git`
- `cd notus-model-demo`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

Next, try one of the demos:

- Streaming: `python stream_response.py`
- Non-streaming: `python generate_response.py`
