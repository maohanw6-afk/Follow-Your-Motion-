# Follow-Your-Motion: Video Motion Transfer via Efficient Spatial-Temporal Decoupled Finetuning

## 📺  Demo Video

https://github.com/user-attachments/assets/6677745d-b869-43b2-8da1-0e4d3ec6564c


## 📖 Introduction

This project is an advanced enhancement based on the **Wan2.1** foundation model, specifically optimized for **motion-following** tasks. 

By training custom **Temporal LoRAs** and **Spatial LoRAs** on input video sequences, the model learns to capture precise motion dynamics and spatial structures. Once the weights are obtained, our inference pipeline generates new videos that faithfully "follow" the movements of the reference source.
# Follow-Your-Motion: Video Motion Transfer via Efficient Spatial-Temporal Decoupled Finetuning

## 📺  Demo Video


https://github.com/user-attachments/assets/48943dfa-2a54-4d16-a5f7-a290290ff2c3



## 📖 Introduction

This project is an advanced enhancement based on the **Wan2.1** foundation model, specifically optimized for **motion-following** tasks. 

By training custom **Temporal LoRAs** and **Spatial LoRAs** on input video sequences, the model learns to capture precise motion dynamics and spatial structures. Once the weights are obtained, our inference pipeline generates new videos that faithfully "follow" the movements of the reference source.


## 🚀 Key Features

* **Dual-LoRA Training**: Support for training both Temporal and Spatial LoRAs to achieve superior motion fidelity.
* **Motion Transfer**: Seamlessly transfer complex actions from reference videos to your generated content.
* **Optimized for Consumer GPUs**: We highly recommend using the **Wan2.1-T2V-1.3B** model, which is optimized for efficiency without sacrificing quality.
* **Memory Efficiency**: Integrated with **Tiled VAE** technology and memory-offloading strategies.


## ⚙️ Setup Environment
```bash
# Create Conda Environment
conda create -n Fym python=3.10
activate Fym

```

## Requirements
 · Python 3.10+
 
 · Pytorch 2.0+
 
 · CUDA 12.x


## Using dowload script
 # Download 1.3B model
```bash
python examples/download_model.py --1.3b
```

 
## 📦 Installation & Setup

### 1. Prerequisites
- **Recommended Model**: **Wan2.1-T2V-1.3B** (Best balance between performance and VRAM usage).
- **GPU**: NVIDIA RTX 3090/4090 (Recommended for training).
- **VRAM**: ~8.2GB for 1.3B inference, 24GB for full LoRA training.

### 2. Environment Configuration
Clone the repository and install the dependencies:
```bash
# Clone this repo
git clone [https://github.com/maohanw6-afk/Follow-Your-Motion.git](https://github.com/maohanw6-afk/Follow-Your-Motion.git)
cd Follow-Your-Motion

# Install required packages
pip install -r requirements.txt
```
💻 Usage Guide



## Step 2: Prepare Your Dataset

Before processing the data, you need to organize your training dataset in the following structure:

```
data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
```

The `metadata.csv` file should contain the file names and their corresponding text descriptions:

```
file_name,text
video_00001.mp4,"video description"
```

## Step 3: Process Your Dataset

This step collects attention map information from your dataset:

```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/EffiVMT_train_wan_t2v.py \
  --task data_process \
  --dataset_path examples/wanvideo/data/example_dataset \
  --output_path ./models \
  --text_encoder_path "examples/wanvideo/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "examples/wanvideo/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 45 \
  --height 544 \
  --width 544
```
## Step 4: Collect Attention Head Information
```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/Collect_attn_map.py \
  --dataset_path examples/wanvideo/data/example_dataset \
  --dit_path "examples/wanvideo/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
```
## Step 5: Train Spatial LoRA

Train the spatial attention components of the model:

```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/EffiVMT_train_wan_t2v_Head.py \
  --task train \
  --train_architecture lora \
  --dataset_path examples/wanvideo/data/example_dataset \
  --output_path models/wan_video \
  --out_file_name='lora_spatial' \
  --dit_path "examples/wanvideo/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 3 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "self_attn.q_spatial,self_attn.k_spatial,self_attn.v_spatial" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --spatial_lora_wd 0.1
```

## Step 6: Train Temporal LoRA

After training the spatial components, train the temporal attention components:

```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/EffiVMT_train_wan_t2v_Head.py \
  --task train \
  --train_architecture lora \
  --dataset_path examples/wanvideo/data/example_dataset \
  --out_file_name='lora_temporal' \
  --dit_path "examples/wanvideo/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 3 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "self_attn.q_temporal,self_attn.k_temporal,self_attn.v_temporal" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --train_temporal_lora \
  --pretrained_spatial_lora_path=models/wan_video/lora_spatial.ckpt \
  --temporal_lora_wd 0.5
```
## 📁 Project Structure

<details>
<summary><b>Click for directory structure</b></summary>

```text
Follow-Your-Motion/
├── diffsynth/                  # Core library
│   ├── models/                 # Model implementations
│   │   ├── wan_video_dit.py    # Modified DiT with Q/K extraction
│   │   ├── wan_video_vae.py    # Video VAE encoder/decoder
│   │   └── wan_video_text_encoder.py
│   ├── pipelines/              # Inference pipelines
│   │   └── wan_video.py        # Pipeline with AMF implementation
│   ├── schedulers/             # Noise schedulers (Flow Matching)
│   ├── prompters/              # Prompt processing
│   └── vram_management/        # Memory optimization utilities
├── examples/                   # Example scripts
│   └── data/example_dataset/
│       ├── metadata.csv
│       └── train/
│           └── video_00001.mp4
├── models/                     # Model checkpoints
├── requirements.txt            # Dependencies
└── setup.py                    # Package setup
```

## 💡 Notes

* **Single-GPU Support**: This project is **single-GPU compatible**, allowing you to run the entire training and inference pipeline on a single graphics card.
* **Two-Phase Training**: The training process is systematically split into two phases: **Spatial Attention** training and **Temporal Attention** training.
* **Hardware Configuration**: Adjust the GPU device index in `CUDA_VISIBLE_DEVICES` as needed to match your local setup.
* **Parameter Tuning**: 
    * You can modify general parameters such as **learning rate**, **number of epochs**, and **LoRA settings** based on your specific needs.
    * Specifically, feel free to adjust `max_epochs`, `spatial_lora_wd`, and `temporal_lora_wd` to optimize performance according to your hardware constraints and video content.
* **Checkpoint Saving**: All trained model checkpoints will be automatically saved to the output directories specified in your configuration.





## 🚀 Key Features

* **Dual-LoRA Training**: Support for training both Temporal and Spatial LoRAs to achieve superior motion fidelity.
* **Motion Transfer**: Seamlessly transfer complex actions from reference videos to your generated content.
* **Optimized for Consumer GPUs**: We highly recommend using the **Wan2.1-T2V-1.3B** model, which is optimized for efficiency without sacrificing quality.
* **Memory Efficiency**: Integrated with **Tiled VAE** technology and memory-offloading strategies.


## ⚙️ Setup Environment
```bash
# Create Conda Environment
conda create -n Fym python=3.10
activate Fym

```

## Requirements
 · Python 3.10+
 · Pytorch 2.0+
 · CUDA 12.x


##Using dowload script
 # Download 1.3B model
```bash
python examples/download_model.py --1.3b
```

 
## 📦 Installation & Setup

### 1. Prerequisites
- **Recommended Model**: **Wan2.1-T2V-1.3B** (Best balance between performance and VRAM usage).
- **GPU**: NVIDIA RTX 3090/4090 (Recommended for training).
- **VRAM**: ~8.2GB for 1.3B inference, 24GB for full LoRA training.

### 2. Environment Configuration
Clone the repository and install the dependencies:
```bash
# Clone this repo
git clone [https://github.com/maohanw6-afk/Follow-Your-Motion.git](https://github.com/maohanw6-afk/Follow-Your-Motion.git)
cd Follow-Your-Motion

# Install required packages
pip install -r requirements.txt
```
💻 Usage Guide



## Step 2: Prepare Your Dataset

Before processing the data, you need to organize your training dataset in the following structure:

```
data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
    └── image_00002.jpg
```

The `metadata.csv` file should contain the file names and their corresponding text descriptions:

```
file_name,text
video_00001.mp4,"video description"
image_00002.jpg,"video description"
```

## Step 3: Process Your Dataset

This step collects attention map information from your dataset:

```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/EffiVMT_train_wan_t2v.py \
  --task data_process \
  --dataset_path examples/wanvideo/data/example_dataset \
  --output_path ./models \
  --text_encoder_path "examples/wanvideo/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "examples/wanvideo/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 45 \
  --height 544 \
  --width 544
```
## Step 4: Collect Attention Head Information
```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/Collect_attn_map.py \
  --dataset_path examples/wanvideo/data/example_dataset \
  --dit_path "examples/wanvideo/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
```
## Step 5: Train Spatial LoRA

Train the spatial attention components of the model:

```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/EffiVMT_train_wan_t2v_Head.py \
  --task train \
  --train_architecture lora \
  --dataset_path examples/wanvideo/data/example_dataset \
  --output_path models/wan_video \
  --out_file_name='lora_spatial' \
  --dit_path "examples/wanvideo/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 3 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "self_attn.q_spatial,self_attn.k_spatial,self_attn.v_spatial" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --spatial_lora_wd 0.1
```

## Step 6: Train Temporal LoRA

After training the spatial components, train the temporal attention components:

```bash
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/EffiVMT_train_wan_t2v_Head.py \
  --task train \
  --train_architecture lora \
  --dataset_path examples/wanvideo/data/example_dataset \
  --out_file_name='lora_temporal' \
  --dit_path "examples/wanvideo/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 3 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "self_attn.q_temporal,self_attn.k_temporal,self_attn.v_temporal" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --train_temporal_lora \
  --pretrained_spatial_lora_path=models/wan_video/lora_spatial.ckpt \
  --temporal_lora_wd 0.5
```

## 💡 Notes

* **Single-GPU Support**: This project is **single-GPU compatible**, allowing you to run the entire training and inference pipeline on a single graphics card.
* **Two-Phase Training**: The training process is systematically split into two phases: **Spatial Attention** training and **Temporal Attention** training.
* **Hardware Configuration**: Adjust the GPU device index in `CUDA_VISIBLE_DEVICES` as needed to match your local setup.
* **Parameter Tuning**: 
    * You can modify general parameters such as **learning rate**, **number of epochs**, and **LoRA settings** based on your specific needs.
    * Specifically, feel free to adjust `max_epochs`, `spatial_lora_wd`, and `temporal_lora_wd` to optimize performance according to your hardware constraints and video content.
* **Checkpoint Saving**: All trained model checkpoints will be automatically saved to the output directories specified in your configuration.

---

