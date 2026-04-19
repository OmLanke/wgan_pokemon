# AI Agent Guidelines for Pokemon DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate new Pokemon images from a dataset of 64x64 pixel Pokemon sprites.

## Project Architecture

### Core Components

| File                   | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| **main.py**            | Entry point for training/inference with CLI flags for hyperparameters  |
| **model.py**           | DCGAN class defining generator and discriminator network architectures |
| **ops.py**             | Custom TensorFlow operations (conv2d, deconv2d, lrelu, batch_norm)     |
| **utils.py**           | Image processing utilities (imread, transform, save_images, merge)     |
| **augmentation.ipynb** | Data augmentation pipeline (flipping, rotation)                        |

### Directory Structure

- `checkpoint/pokemon_64_64_64/` - Pre-trained model checkpoints (TensorFlow 1.x format)
- `data/pokemon/` - Training dataset location
- `samples/` - Generated output images during training

## Quick Start

### Training

```bash
python main.py --dataset pokemon --train --epoch 2000
```

### Inference from Pre-trained Model

```bash
python main.py --dataset pokemon --generate_test_images 100
```

### Key CLI Flags

- `--dataset pokemon` - Required for pokemon dataset
- `--train` - Enable training mode
- `--epoch 2000` - Number of training epochs
- `--batch_size 64` - Batch size (default 64)
- `--learning_rate 0.0002` - Adam optimizer learning rate
- `--output_height 64` / `--output_width 64` - Image dimensions
- `--checkpoint_dir` - Path to save/load checkpoints
- `--sample_dir` - Path to save generated samples

## Technical Details

### DCGAN Architecture

**Generator:**

- Input: 100-dim noise vector (uniform distribution)
- 5 deconvolution layers with ReLU activations
- Output: 3×64×64 RGB image (Tanh activation)

**Discriminator:**

- Input: 3×64×64 RGB image
- 5 convolution layers with Leaky ReLU activations
- Output: Binary classification (Sigmoid)

### Data Format

- Input images: 64×64 pixels, RGB (3-channel)
- PNG with transparency: Converted to JPG with white background
- Data augmentation: Horizontal flip, ±3°/±5°/±7° rotation

### Training Hyperparameters

- Batch size: 64
- Learning rate: 0.0002 (Adam optimizer)
- Momentum (beta1): 0.5
- Weight initialization: Normal distribution (std=0.02)
- Leaky ReLU slope: 0.2

## Important Notes for Agents

### Legacy Code

This is TensorFlow 1.x code (circa 2017). When modifying:

- Use TensorFlow 1.x APIs - compatibility shims exist in `ops.py` for older versions
- Checkpoints use TensorFlow 1.x SaverFormat - not compatible with TensorFlow 2.x
- `tf.app.flags` is TensorFlow 1.x style (deprecated in 2.x)

### Data Dependencies

- Training requires `data/pokemon/` directory with Pokemon images
- Pre-trained checkpoints exist but assume specific architecture
- Image preprocessing done via `utils.get_image()` - respects crop/resize settings

### Common Development Tasks

1. **Modify network architecture**: Edit generator/discriminator in `model.py`
2. **Adjust training hyperparameters**: Use CLI flags in `main.py` or modify defaults
3. **Add data augmentation**: Update `augmentation.ipynb` and re-run preprocessing
4. **Analyze training**: Loss curves logged during training, samples saved to `sample_dir`

## References

- See [README.md](README.md) for detailed DCGAN theory and experimental results
- Original architecture from [@carpedm20](https://github.com/carpedm20/DCGAN-tensorflow)
