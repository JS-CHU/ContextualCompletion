# Digging into Intrinsic Contextual Information for High-fidelity 3D Point Cloud Completion

<div align="center">

</div>
</div>
<p align="center">
  <a href='https://arxiv.org/abs/2412.08326'><img src='https://img.shields.io/badge/Arxiv-2412.08326-b31b1b.svg?logo=arXiv'></a>
</p>

The code repository implemented in PyTorch for paper [Digging into Intrinsic Contextual Information for High-fidelity 3D Point Cloud Completion](https://arxiv.org/abs/2412.08326) (AAAI 2025).

## Requirements

Our model have been tested on the configuration below:

- python == 3.10
- PyTorch == 2.0.1
- CUDA == 11.8
- PyTorch3d == 0.7.5
- open3d == 0.18.0

Create and activate conda environment:

```
conda env create -n ContextualCompletion -f environment.yml
conda activate ContextualCompletion
```

## Datasets Preparation

First, please download the [ShapeNetViPC-Dataset](https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH28A) (143GB, code: **ar8l**). Then run `cat ShapeNetViPC-Dataset.tar.gz* | tar zx`, you will get `ShapeNetViPC-Dataset` contains three floders: `ShapeNetViPC-Partial`, `ShapeNetViPC-GT` and `ShapeNetViPC-View`.

For each object, the dataset include partial point cloud (`ShapeNetViPC-Patial`), complete point cloud (`ShapeNetViPC-GT`) and corresponding images (`ShapeNetViPC-View`) from 24 different views. You can find the detail of 24 cameras view in `/ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt`.

Run`unzip train_test_list.zip` to get train and test list.

## Run the code

### Train DCG

Run:

```
python train_DCG.py
```

The training logs and model weights will be in logs_results/logs_DCG

### Train CRef

Run:

```
python train_CRef.py
```

The training logs and model weights will be in logs_results/logs_CRef

### Test and Evaluation

```
python test.py
```

The completion results will be in logs_results/results_completion

## Citation

```
@article{chu2024digging,
  title={Digging into Intrinsic Contextual Information for High-fidelity 3D Point Cloud Completion},
  author={Chu, Jisheng and Li, Wenrui and Wang, Xingtao and Ning, Kanglin and Lu, Yidan and Fan, Xiaopeng},
  journal={arXiv preprint arXiv:2412.08326},
  year={2024}
}
```