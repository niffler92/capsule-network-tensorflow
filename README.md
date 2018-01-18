# capsule-network-tensorflow
Tensorflow implementation of Capsule Network using slim and recently updated Dataset API.  
You can achieve validation accuracy up to ~99.62%.  


`python download_data.py --dataset mnist`  
`python train.py --dataset mnist`  


# Training Results
<center>
    <img src="https://github.com/niffler92/capsule-network-tensorflow/blob/master/assets/training_summary.PNG" width="900", height="300")
<center/>

<center>
    <img src="https://github.com/niffler92/capsule-network-tensorflow/blob/master/assets/reconstructed_image.PNG" width="900", height="300")
<center/>

# Reference
* [Dynamic Routing Between Capsules](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)
* [Tensorflow implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow)
