<div align="center">
    <h1>🤖 OccRWKV</h1>
    <h2>Rethinking Efficient 3D Semantic Occupancy Prediction with Linear Complexity</h2> <br>
     <a href='https://jmwang0117.github.io/OccRWKV/'>Project_Page</a>
</div>


## 📢 News

- [2024/09]: OccRWKV's logs are available for download:
<div align="center">

| OccRWKV Results | Experiment Log |
|:------------------------------------------------------------------:|:----------:|
|OccRWKV on the SemanticKITTI hidden official test dataset | [link](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3009632_connect_hku_hk/EYqFDMD6xexCqXwfZ_nPxEUB0akfqePg4TwuGiuf4fQK0Q?e=PFM1ma) |
|OccRWKV train log | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/EcKG5MgDCTJJuu8DJ7VoS9sB0euzAEaMkpLjlY9LvRJ0GA?e=lwddX3) |

</div>

- [2024/08]: The pre-trained model can be downloaded at  [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/ETCUIJ7rPnFJniQYMsDsPyIBHkzirRP4c3n-eU9fcBZTaA?e=P8AkQ2).
- [2024/07]: 🔥 We released the code of OccRWKV. *The First Receptance Weighted Key Value (RWKV)-based 3D Semantic Occupancy Network*

</br>

```
@article{wang2024omega,
title={OccRWKV: Rethinking Efficient 3D Semantic Occupancy Prediction with Linear Complexity},
author={Wang, Junming and Yin, Wei and Long, Xiaoxiao and Zhang, Xinyu and Xing, Zebing and Guo, Xiaoyang and Qian, Zhang},
year={2024}
      } 
```

Please kindly star ⭐️ this project if it helps you. We take great efforts to develop and maintain it 😁.


## 🛠️ Installation

```
conda create -n occ_rwkv python=3.10 -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install spconv-cu120
pip install tensorboardX
pip install dropblock
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -U openmim
mim install mmcv-full
pip install mmcls==0.25.0
```

## 💽 Dataset

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

## 🤗 Getting Start
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
$ bash scripts/run_train.sh
```
### Validation

Validation passes are done during training routine. Additional pass in the validation set with saved model can be done by using the `validate.py` file. You need to provide the path to the saved model and the dataset root directory.

```
$ cd <root dir of this repo>
$ bash scripts/run_val.sh
```
### Test

Since SemantiKITTI contains a hidden test set, we provide test routine to save predicted output in same format of SemantiKITTI, which can be compressed and uploaded to the [SemanticKITTI Semantic Scene Completion Benchmark](http://www.semantic-kitti.org/tasks.html#ssc).

We recommend to pass compressed data through official checking script provided in the [SemanticKITTI Development Kit](http://www.semantic-kitti.org/resources.html#devkit) to avoid any issue.

You can provide which checkpoints you want to use for testing. We used the ones that performed best on the validation set during training. For testing, you can use the following command.

```
$ cd <root dir of this repo>
$ bash scripts/run_test.sh
```


## 🏆 Acknowledgement
Many thanks to these excellent open source projects:
- [AGRNav](https://github.com/jmwang0117/AGRNav)
- [Prometheus](https://github.com/amov-lab/Prometheus)
- [SSC-RS](https://github.com/Jieqianyu/SSC-RS)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
- [Terrestrial-Aerial-Navigation](https://github.com/ZJU-FAST-Lab/Terrestrial-Aerial-Navigation)
- [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)
