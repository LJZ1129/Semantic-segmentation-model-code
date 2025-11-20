# Semantic-segmentation-model-code
# Project Title: [Research on a Portable UAV-Based Method for Information Acquisition and Field Lithological Identification via Semantic Segmentation]
This repository contains the implementation of semantic segmentation models for lithology identification, as presented in the paper "Research on a Portable UAV-Based Method for Information Acquisition and Field Lithological Identification via Semantic Segmentation".
# Background
Lithology identification in high-altitude and deeply dissected terrains poses significant challenges to traditional geological mapping. This project aims to leverage UAV remote sensing and deep learning to provide an efficient and accurate solution for this problem.
# Key Contributions
* Constructed a novel CA-DeepLabV3+ model for high-resolution lithological mapping.
* Achieved significant performance improvements (e.g., 97.95% OA, 95.71% mIoU) over baseline models in complex geological environments.
* Demonstrated the model's capability for automatic error correction in lithological interpretation through field validation.
# Model Overview
The core of this project is the CA-DeepLabV3+ model, an enhancement of the DeepLabV3+ semantic segmentation framework. It integrates a Coordinate Attention (CA) mechanism and a multi-scale feature fusion module with a lightweight MobileNetV2 backbone to improve spatial positional encoding and fine-scale feature extraction.
# File Structure
.
├── CA-DeepLabV3+ Model.py    # Implementation of the proposed CA-DeepLabV3+ model
├── DeepLabV1 Model.py        # Implementation of DeepLabV1 for comparison
├── DeepLabV3+ Model.py       # Implementation of standard DeepLabV3+ for comparison
├── FCN Model.py              # Implementation of FCN for comparison
├── FPN Model.py              # Implementation of FPN for comparison
├── PSPNet Model.py           # Implementation of PSPNet for comparison
├── U-Net Model.py            # Implementation of U-Net for comparison
├── README.md                 # This README file
└── dataset/
    ├── train/                # Used for training models
    ├── val/                  # Used for validation and evaluation after model training
    └── test/                 # Used to demonstrate the model's generalization ability
        ├── images/           # Sample raw images
        └── labels/           # Corresponding ground truth masks
# Dataset
The full dataset for lithology identification, comprising sandstone, diorite, marble, and Quaternary sediments, was generated from UAV orthophotos of the Ququleke region. Due to the large size of the complete dataset, it cannot be fully uploaded. However, a small sample dataset (e.g., 10 images with their corresponding labels) can be found in the 'dataset/' directory to facilitate code testing.
# Environment Setup
The experiments were conducted on a Windows 11 operating system. A virtual working environment was configured using the Anaconda environment manager with Python 3.12.3 and PyTorch 2.6.0. The hardware setup comprised an Intel Core i9‑13900HX CPU and an NVIDIA RTX 4060 GPU. The initial learning rate was set to 0.001, with the AdamW optimizer and AMP mixed-precision training employed. Due to limited computational resources, the batch size was set to 8, the number of epochs to 100, and the patch size to 256×256, with five target classes in total.
# Contact
For any questions or suggestions, feel free to contact Jingzhi Liu at [3411463934@qq.com].
