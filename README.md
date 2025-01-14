# Adventitious Heart and Lung Sounds Classification to Diagnose Diseases Using AI

### This is the Repository for the Group Project: GP8000 - Artificial Intelligence Literacy

## Introduction 
The Overview of the proposed algorithm.
<p align="center">
  <img src="./asset/ICBHI.png" alt="drawing", width="650"/>
</p>
Our project aims to enhance diagnostic accuracy by classifying abnormal heart and lung sounds. 
Utilizing the open-source heart and lung sound databases, we implemented Multi-Task Network to 
detect and categorize specific adventitious sounds associated with various respiratory and 
cardiovascular conditions. This approach leverages advanced algorithms to process and analyze 
audio signals, helping to automate and improve disease diagnosis while potentially reducing the 
need for manual auscultation. Through this project, we seek to demonstrate the potential of 
AI in advancing non-invasive diagnostic methods for clinical applications.

## Environment
Training and evaluation environment: Python 3.12.0, PyTorch 2.4.1, CUDA 12.1. 

Run the following command to create the 
conda environment and install required packages.

```
conda create -n LungHeart python==3.12.0 -y
conda activate LungHeart
pip3 install torch torchvision torchaudio
pip3 install tqdm
pip3 install wandb
pip3 install pandas
pip3 install scikit-learn
pip3 install transformers
pip3 install albumentations
pip3 install fvcore
```

## Dataset

The training, validating and testing dataset are both subset from ICBHI 2017.

Download the dataset from: [ICBHI]: https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip

Arrange the dataset folder as follows:

```
ICBHI Dataset
│ 
├── official_split.txt          # Official data split 
├── metadata.txt                # Basic information about each patient
├── filename_format.txt         # File naming convention
├── filename_differences.txt    # File name discrepancies
├── *.txt/                      # Each sample’s diagnostic label
└── *.wav/                      # Audio files
```

## Training

Before training, please download the datasets and complete environment configuration, and then configure the model in `./config/ICBHIModel.yaml`.

Use the following code to start the training process. Suppose the root path of dataset is '/dataset/ICBHI/'

```
python main.py \
    --exp_name=FATModel \
    --model_cfg='./config/ICBHIModel.yaml' \
    --data_root='/dataset/ICBHI/' \
    --batch_size=16 \
    --gpu=0 \
    --save_path=./experiments/ \
    --vis 
```

The testing result will be saved at './experiments/save/'.

The best checkpoint will be saved at './experiments/ckpt/'.

## Testing

Before testing, please download the datasets and complete environment configuration, and use the default configuration in `./config/ICBHIModel.yaml`.

Use the following code to start the testing process. Suppose the root path of dataset is '/dataset/ICBHI/'.

```
python main.py \
    --exp_name=FATModel \
    --model_cfg='./config/TransNeXtModel.yaml' \
    --data_root='/dataset/celebA/' \
    --batch_size=16 \
    --gpu=0 \
    --save_path='./experiments/' \
    --vis \
    --test \
    --ckpt_path='./best_ckpt.pth'
```

The testing result will be saved at './experiments/save/'.
The F measure result and the computational analysis result will be print on the screen.

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source.

