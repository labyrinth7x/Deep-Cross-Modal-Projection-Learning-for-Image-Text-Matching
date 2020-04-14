# Deep Cross-Modal Projection Learning for Image-Text Matching
This is a Pytorch implmentation for the paper [Deep Cross-Modal Projection Learning for Image-Text Matching](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ying_Zhang_Deep_Cross-Modal_Projection_ECCV_2018_paper.pdf).   
The official implementation in TensorFlow can be found [here](https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning).   
## Requirement   
* Python 3.5 
* Pytorch 1.0.0 & torchvision 0.2.1
* numpy
* scipy 1.2.1 

## Data Preparation
- Download the pre-computed/pre-extracted data from [GoogleDrive](https://drive.google.com/drive/folders/1Nbx5Oa5746_uAcuRi73DmuhQhrxvrAc9?usp=sharing) and move them to ```data/processed``` folder. Or you can use the file ```dataset/preprocess.py``` to prepare your own data.
- *[Optional]* Download the pre-trained model weights from [GoogleDrive](https://drive.google.com/drive/folders/1LtTjWeGuLNvQYMTjdrYbdVjbxr7bLQQC?usp=sharing) and move them to ```pretrained_models``` folder.

## Training & Testing
You should firstly change the param ```model_path``` to your current directory.   
```
sh scripts/run.sh
```
You can directly run the code instead of performing training and testing seperately.   
Or training:  
```
sh scripts/train.sh  
```
Or testing:
```
sh scripts/test.sh  
```
