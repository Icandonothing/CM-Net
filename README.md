# CM-Net

## Architecture

<p align="center">
<img src="module.png" width=100% height=40% 
class="center">
</p>

## Qualitative Results

<p align="center">
<img src="results.png" width=100% height=40% 
class="center">
</p>

## Usage:
### Recommended environment:
```
Python 3.8
Pytorch 1.12.1
torchvision 0.13.1
```
### Data preparation:
- **Polyp datasets:**
Download the training 和 testing datasets [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing) and move them into './data/polyp/' folder.

### Pretrained model:
You should download the pretrained PVTv2 model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing).

## Acknowledgement
We are very grateful for these excellent works [timm](https://github.com/huggingface/pytorch-image-models), [MERIT](https://github.com/SLDGroup/MERIT), [CASCADE](https://github.com/SLDGroup/CASCADE), [PraNet](https://github.com/DengPingFan/PraNet), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and [TransUNet](https://github.com/Beckschen/TransUNet), which have provided the basis for our framework.

## Citations
```
@inproceedings{rahman2024g,
  title={G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation},
  author={Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7728--7737},
  year={2024}
}
```

