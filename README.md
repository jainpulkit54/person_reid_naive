# person_reid_naive

A triplet network takes in three images as input i.e., an anchor image, a positive image (i.e., image having label same as the anchor) and a negative image (i.e., image having label different from the anchor). The objective here is to learn embeddings such that the positive images are closer to the anchor as compared to the negative images. The same can be pictorically represented using the image given below:<br>
<img src = "images/anchor_negative_positive.png"></img>

Source: <a href = "https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf">Schroff, Florian, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. CVPR 2015</a><br>

Moreover, the triplet loss is mathematically expressed as:<br>
<img src = "images/triplet_loss.png"></img>

### Dataset

The dataset on which this model has been trained is <b>Market-1501</b>. Dataset description can be found on <a href = "https://www.aitribune.com/dataset/2018051063">this link</a>. The dataset can be downloaded from the kaggle website.

### Network Architecture and Data Augmentation
The network architecture used is:<br>
<i>Pretrained ResNet-50 > Linear 1024 > BatchNorm > ReLU > Linear 128</i><br>
The dataset is augmented on the go (during training) by using Random Horizontal Flips.<br>

### Tensorboard Visualization

The training logs obtained are as follows:<br>
<img src = "images/logs_naive_triplet_loss.png"></img>

Moreover the training logs can be visualized by following instructions as:<br>
1) Go to the source directory.<br>
2) Type the command:<br>
<code>$ tensorboard --logdir logs_market1501_naive</code><br>
3) Go to a browser and type:<br>
<code>http://localhost:6006/</code>

### Performance Evaluation
