# TRACER
TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object  Refinement


https://github.com/user-attachments/assets/1cc95a65-1163-4590-bdab-cd51c69770f4


## Usage 

### 1.Requirements   

Code is tested under Pytorch 1.12.1, python 3.7, and CUDA 11.3 

```
pip install -r requirements.txt
```

### 2.Dataset   

Download the Fine-AGDDO15 dataset from [Baidu Pan](https://pan.baidu.com/s/1bD4HMDlZnyQaCKut3CE3Fg?pwd=D5FS)[D5FS].(you can annotate your own one-shot data in the same format).  

Put the data in the dataset folder with the following structure:  

```
dataset 
├── one-shot-seen
└── Seen
```

### 3.Train and Test   Run following commands to start training or testing:
```
python train.py
python test.py --model_file <PATH_TO_MODEL>
```
