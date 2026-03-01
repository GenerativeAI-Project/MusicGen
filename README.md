# MusicGen Baseline: Conditional Genre Music Generation

Baseline implementation for a conditional musical GenAI model capable of generating music samples in a specific genre. Built on Meta's [AudioCraft](https://github.com/facebookresearch/audiocraft) library and the [Free Music Archive](https://huggingface.co/datasets/benjamin-paine/free-music-archive-medium) dataset.

## Project Overview

- **Model**: `facebook/musicgen-small` (300M parameters) via AudioCraft
- **Dataset**: Free Music Archive Medium (24,801 tracks, 30s clips, 16 genres)
- **Evaluation**: Fréchet Audio Distance (FAD)
- **Baseline Sampling**: Standard Ancestral Sampling

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch 2.1+, and FFmpeg (recommended).

## Project Structure

```
MusicGen/
├── src/
│   ├── data/
│   │   └── pipeline.py      # FMA dataset loading & manifest generation
│   ├── generate.py         # Music generation with ancestral sampling
│   └── evaluate.py         # FAD evaluation
├── config/
│   └── default.yaml       # Default configuration
├── COMMIT_TIMELINE.md     # Commit schedule for Daria & Kamilya
├── requirements.txt
└── README.md
```

## Usage

### 1. Prepare Data (Dataset Pipeline)

Generate AudioCraft-compatible manifests from FMA:

```bash
python -m src.data.pipeline --output_dir ./data/manifests --split_ratio 0.9 0.05 0.05
```

### 2. Generate Music

Generate samples using ancestral sampling (baseline):

```bash
python -m src.generate --prompts "upbeat electronic dance music" --output_dir ./outputs --num_samples 10
```

### 3. Evaluate with FAD

Compute Fréchet Audio Distance between generated and reference audio:

```bash
python -m src.evaluate --background_dir ./data/reference --eval_dir ./outputs --output_file ./results/fad_score.json
```

## Configuration

Edit `config/default.yaml` for model, generation, and evaluation parameters.

## References

- [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) (Meta AI, 2024)
- [Fréchet Audio Distance](https://arxiv.org/abs/1812.08466) (Google AI, 2019)
- [FMA Dataset](https://arxiv.org/abs/1612.01840) (ISMIR, 2017)
