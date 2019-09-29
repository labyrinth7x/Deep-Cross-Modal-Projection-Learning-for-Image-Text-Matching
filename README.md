# Deep Cross-Modal Projection Learning for Image-Text Matching
This is a simple implmentation for the paper [Deep Cross-Modal Projection Learning for Image-Text Matching](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ying_Zhang_Deep_Cross-Modal_Projection_ECCV_2018_paper.pdf)<br>
The official implementation is by TensorFlow.[here](https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning)<br>
## Requirement<br>
* Python 3.5 
* Pytorch 1.0.0 & torchvision 0.2.1
* numpy
* matplotlib (not necessary unless the need for the result figure)  
* scipy 1.2.1 
## Network<br>
The backbone of the network is MobileNet.<br>
It is implemented in ./models/mobilenet.py.<br>
## Train & Test 
```
sh scripts/run.sh
```
