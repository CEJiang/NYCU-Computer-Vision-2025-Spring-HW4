# NYCU Computer Vision 2025 Spring HW4
- StudentID: 313553044
- Name: 江仲恩

## Introduction
In this assignment, we address the task of image restoration under challenging degradation conditions, such as synthetic rain and snow. Given the complexity of natural corruption patterns, a robust and effective model is essential for accurately recovering image details. To this end, we adopt the PromptIR All-in-One architecture, which integrates prompt-based feature modulation with transformer-based encoding to enhance restoration across multiple degradation types.

Our model is specifically designed and optimized to perform well on the HW4 image restoration dataset, where each image is paired with a clean ground truth. PromptIR employs a multi-level prompt injection mechanism, enabling the network to adapt its feature processing dynamically, which greatly improves restoration quality by preserving structural information and reducing residual artifacts.

To further enhance performance, we incorporate **Test-Time Augmentation (TTA)** during inference and use a hybrid training loss that combines **L1 and SSIM**, striking a balance between pixel-wise accuracy and perceptual quality. Experimental results show that our method achieves **31.17 dB** PSNR on the public leaderboard, demonstrating its effectiveness in restoring realistic visual content under synthetic degradations.

## How to install

1. Clone the repository
```
git clone https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW4.git
cd NYCU-Computer-Vision-2025-Spring-HW4
```

2. Create and activate conda environment
```
conda env create -f environment.yml
conda activate cv
```

3. Download the dataset 
- You can download the dataset from the provided [LINK](https://drive.google.com/file/d/1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w)
- Place it in the following structure
```
NYCU-Computer-Vision-2025-Spring-HW4
├── hw4_release_dataset
│   ├── train
│   └── test
├── environment.yml
├── main.py
├── train.py
├── test.py
.
.
.
```

4. Run for Train
    1. Train Model 
    ```
    python main.py DATAPATH [--epochs EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--decay DECAY] [--eta_min ETA_MIN] [--saved_path SAVE_FOLDER] [--mode train]
    ```
    Example
    ```
    python main.py ./hw4_release_dataset --epochs 80 --batch_size 1 --learning_rate 1e-4 --decay 5e-3 --saved_path saved_models
    ```
    2. Test Model
    ```
    python main.py DATAPATH --mode test
    ```
    Example
    ```
    python main.py ./hw4_release_dataset --mode test
    ```

## Performance snapshot
### Training Parameter Configuration

| Parameter        | Value                                                                                                   |
|------------------|---------------------------------------------------------------------------------------------------------|
| Pretrained Weight| None                                                                                                    |
| Learning Rate    | 0.0001                                                                                                  |
| Batch Size       | 1                                                                                                       |
| Epochs           | 80                                                                                                      |
| decay            | 0.005                                                                                                   |
| Optimizer        | AdamW                                                                                                   |
| Eta_min          | 0.000001                                                                                                |
| T_max            | 80                                                                                                      |
| Scheduler        | `CosineAnnealingLR`                                                                                     |
| ratio            | `0.3` -> `0.7`                                                                                          |
| Criterion        | `(1 - ratio) * L1 Loss` + `ratio * SSIM Loss`                                                           |

### Training Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW4/blob/main/Image/training_curve.png)
### PSNR Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW4/blob/main/Image/psnr_curve.png)

### Performance
|                  | mAP                      |
|------------------|--------------------------|
| Validation       | 30.32                    |
| Public Test      | 31.17                    |
