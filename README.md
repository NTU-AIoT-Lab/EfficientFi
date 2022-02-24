# EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression

## Introduction
WiFi technology has been applied to various places due to the increasing requirement of high-speed Internet access. Recently, besides network services, WiFi sensing is appealing in smart homes since it is device-free, cost-effective and privacy-preserving. Though numerous WiFi sensing methods have been developed, most of them only consider single smart home scenario. Without the connection of powerful cloud server and massive users, large-scale WiFi sensing is still difficult. In this paper, we firstly analyze and summarize these obstacles, and propose an efficient large-scale WiFi sensing framework, namely EfficientFi. The EfficientFi works with edge computing at WiFi APs and cloud computing at center servers. It consists of a novel deep neural network that can compress fine-grained WiFi Channel State Information (CSI) at edge, restore CSI at cloud, and perform sensing tasks simultaneously. A quantized auto-encoder and a joint classifier are designed to achieve these goals in an end-to-end fashion. To the best of our knowledge, the EfficientFi is the first IoT-cloud-enabled WiFi sensing framework that significantly reduces communication overhead while realizing sensing tasks accurately. We utilized human activity recognition and identification via WiFi sensing as two case studies, and conduct extensive experiments to evaluate the EfficientFi. The results show that it compresses CSI data from 1.368Mb/s to 0.768Kb/s with extremely low error of data reconstruction and achieves over 98% accuracy for human activity recognition.

![image.png](attachment:image.png)

You can read our [paper](https://doi.org/10.1109/JIOT.2021.3139958) here

## Requirements

```
scipy - 1.7.3
numpy - 1.21.5
torchvision - 0.11.3
pytorch - 1.10.2
```

## Run

To run the EfficientFi, simply run `python main.py`


## Model

The EfficientFi has the following components:

- <font color='green'>***class***</font> <font color='blue'>**Quantize**</font> : find the index of the closest embedding vector and transform the encoder output `z` into the discrete embedding vector `quantized`
- <font color='green'>***class***</font> <font color='blue'>**Encoder**</font> : encode input `x` to `z`
- <font color='green'>***class***</font> <font color='blue'>**Decoder**</font> : rebuild the `quantized` to `r_x`
- <font color='green'>***class***</font> <font color='blue'>**Classifier**</font> : predict the probability of each class in `y_p`

## Reference

@article{yang2022efficientfi,
  title={EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression},
  author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Wang, Dazhuo and Xu, Qianwen and Xie, Lihua},
  journal={IEEE Internet of Things Journal},
  year={2022},
  publisher={IEEE},
  doi={10.1109/JIOT.2021.3139958}
} 
