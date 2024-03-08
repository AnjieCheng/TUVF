# TUVF: Learning Generalizable Texture UV Radiance Fields
[[website]](https://www.anjiecheng.me/TUVF/)
[[paper]](https://www.anjiecheng.me/assets/TUVF/TUVF_compressed.pdf)
[[arxiv]](https://arxiv.org/abs/2305.03040) <br>
<p align="center"><img src="assets/banner.png" width="100%"/></p>
We propose Texture UV Radiance Fields (TUVF), a category-level texture representation disentangled from 3D shapes. Our methods trains from only a collection of real-world images and a set of untextured shapes. Given a 3D shape, TUVF can synthesis realistic, high-fidelity, and diverse 3D consistent textures.

## Code release progress:
- [x] Installation and running instructions
- [x] Datasets
- [x] Training
- [x] Inference code (FID/LPIPS)
- [x] Pre-trained checkpoints
- [x] Editing

## Development Guide

### Environment
We recommend using conda to manage the environment. To create a new environment, run:
```sh
conda create -n tuvf python=3.9.16 -y
conda activate tuvf
```

Then install other dependencies:

```sh
# We prefer to install pytorch at first.
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Then install pytorch3d. This can take several minutes. Make sure that your compilation CUDA version and runtime CUDA version match.
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install other dependencies.
pip install -r requirements.txt
```
Finally, change the `python_bin` setting in the `configs/env` file to match the location of your Python installation. If you're using Conda as mentioned earlier, the path will probably look something like this: `/home/USER/miniconda3/envs/tuvf/bin/python`.


### Datasets
We follow[Texturify](https://github.com/nihalsid/stylegan2-ada-3d-texture)'s dataset format. To download and extract the files for our dataset, please execute the following command. Please ensure that you have the pigz tool installed as it is required for file extraction.
```sh
mkdir CADTextures
cd CADTextures

# CompCars
wget https://huggingface.co/datasets/a8cheng/TUVF/resolve/main/CompCars.tar.gz
tar -I pigz -xvf CompCars.tar.gz -C ./

# PhotoShape Straight
wget https://huggingface.co/datasets/a8cheng/TUVF/resolve/main/Photoshape.tar.gz
tar -I pigz -xvf CompCars.tar.gz -C ./
```
They should follow the folder structure as below:
```
├── CADTextures
│   ├── CompCars
│       ├── exemplars_highres
│       ├── exemplars_highres_mask
│       ├── filelist
│       ├── pretrain
│       ├── shapenet_psr
│   ├── Photoshape
│       ├── straight
│       ├── straight_mask
│       ├── filelist
│       ├── pretrain
│       ├── shapenet_psr
```

## Training

To launch training, run:
```sh
# train TUVF on CompCars
python src/infra/launch.py dataset=compcars dataset.resolution=512 num_gpus=8 training=p128_60_5000 training.batch_size=160 model=canograf exp_suffix=WHATEVER_NAME

# train TUVF on Photoshape
python src/infra/launch.py dataset=photoshape dataset.resolution=512 num_gpus=8 training=p128_60_5000 training.batch_size=160 model=canograf exp_suffix=WHATEVER_NAME
```

To configure the dataset path, you have two options:
1. Modify the dataset path directly in the `configs/dataset/DATASET_NAME` file:
```yaml
path: /YOUR_PATH/CADTextures/DATASET_NAME/
```
2. Alternatively, you can specify the dataset path using a command-line argument:
```sh
dataset.path=YOUR_PATH/CADTextures/DATASET_NAME
```

## Evaluation
To run the evaluation for a checkpoint, you can use the following command. The script performs the following steps: preprocesses real samples, generates synthesized views, and evaluates FID and KID. It then proceeds to evaluate LPIPS_g and LPIPS_t.
```sh
# test TUVF on CompCars
python scripts/evaluate_cars.py ckpt.network_pkl=YOURPATH/CHECKPOINT.pkl dataset_path=YOURPATH/CADTextures/CompCars  output_dir=YOURPATH

# test TUVF on Photoshape
python scripts/evaluate_chairs.py ckpt.network_pkl=YOURPATH/CHECKPOINT.pkl dataset_path=YOURPATH/CADTextures/Photoshape  output_dir=YOURPATH
```

You can also use the following argument to evaluate from our checkpoints
```sh
# CompCars checkpoint
ckpt.network_pkl=https://huggingface.co/datasets/a8cheng/TUVF/resolve/main/checkpoints/cars.pkl

# Photoshape checkpoint
ckpt.network_pkl=https://huggingface.co/datasets/a8cheng/TUVF/resolve/main/checkpoints/chairs.pkl
```

## Editing
```sh
python scripts/finetune.py  demo_dir=/home/anjie/Downloads/CADTextures/finetune_demo dataset_path=/home/anjie/Downloads/CADTextures/CompCars
```

## Acknowledgement

We have used code snippets from different repositories, including the following repositories: [EpiGRAF](https://github.com/universome/epigraf), [NeuMesh](https://github.com/zju3dv/NeuMesh.git), [Texturify](https://github.com/nihalsid/stylegan2-ada-3d-texture), [ShapeAsPoints](https://github.com/autonomousvision/shape_as_points). We would like to acknowledge and thank the authors of these repositories for their excellent work.

## Citation
```
@inproceedings{cheng2023tuvf,
    title     = {TUVF: Learning Generalizable Texture UV Radiance Fields},
    author    = {Cheng, An-Chieh and Li, Xueting and Liu, Sifei and Wang, Xiaolong},
    booktitle = {ICLR},
    year      = {2024}
}
```