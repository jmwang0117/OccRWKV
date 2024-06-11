# OccRWKV

## OccRWKV: Rethinking Sparse Latent Representation for 3D Semantic Occupancy Prediction


## Preperation

### Prerequisites
```
conda create -n occ_rwkv python=3.10 -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install spconv-cu120
pip install tensorboardX
pip install dropblock
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Dataset

Please download the Semantic Scene Completion dataset (v1.1) from the [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html) and extract it.

Or you can use [voxelizer](https://github.com/jbehley/voxelizer) to generate ground truths of semantic scene completion.

The dataset folder should be organized as follows.
```angular2
SemanticKITTI
├── dataset
│   ├── sequences
│   │  ├── 00
│   │  │  ├── labels
│   │  │  ├── velodyne
│   │  │  ├── voxels
│   │  │  ├── [OTHER FILES OR FOLDERS]
│   │  ├── 01
│   │  ├── ... ...
```

## Getting Start
Clone the repository:
```
https://github.com/jmwang0117/OccRWKV.git
```

We provide training routine examples in the `cfgs` folder. Make sure to change the dataset path to your extracted dataset location in such files if you want to use them for training. Additionally, you can change the folder where the performance and states will be stored.
* `config_dict['DATASET']['DATA_ROOT']` should be changed to the root directory of the SemanticKITTI dataset (`/.../SemanticKITTI/dataset/sequences`)
* `config_dict['OUTPUT']['OUT_ROOT'] ` should be changed to desired output folder.

### Train SSC-RS Net

```
$ cd <root dir of this repo>
$ python train.py --cfg cfgs/DSC-Base.yaml --dset_root <path/dataset/root>
```
### Validation

Validation passes are done during training routine. Additional pass in the validation set with saved model can be done by using the `validate.py` file. You need to provide the path to the saved model and the dataset root directory.

```
$ cd <root dir of this repo>
$ python validate.py --weights </path/to/model.pth> --dset_root <path/dataset/root>
```
### Test

Since SemantiKITTI contains a hidden test set, we provide test routine to save predicted output in same format of SemantiKITTI, which can be compressed and uploaded to the [SemanticKITTI Semantic Scene Completion Benchmark](http://www.semantic-kitti.org/tasks.html#ssc).

We recommend to pass compressed data through official checking script provided in the [SemanticKITTI Development Kit](http://www.semantic-kitti.org/resources.html#devkit) to avoid any issue.

You can provide which checkpoints you want to use for testing. We used the ones that performed best on the validation set during training. For testing, you can use the following command.

```
$ cd <root dir of this repo>
$ python test.py --weights </path/to/model.pth> --dset_root <path/dataset/root> --out_path <predictions/output/path>
```
### Pretrained Model

You can download the models with the scores below from this [Google drive link](https://drive.google.com/file/d/1-b3O7QS6hBQIGFTO-7qSG7Zb9kbQuxdO/view?usp=sharing),

| Model  | Segmentation | Completion |
|--|--|--|
| SSC-RS | 24.2 | 59.7 |

<sup>*</sup> Results reported to SemanticKITTI: Semantic Scene Completion leaderboard ([link](https://codalab.lisn.upsaclay.fr/competitions/7170\#results)).

## Acknowledgement
This project is not possible without multiple great opensourced codebases.
* [spconv](https://github.com/traveller59/spconv)
* [LMSCNet](https://github.com/cv-rits/LMSCNet)
* [SSA-SC](https://github.com/jokester-zzz/SSA-SC)
* [GASN](https://github.com/ItIsFriday/PcdSeg)
