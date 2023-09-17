# Universal EHR Federated Learning Framework
This repository is the official implementation for [UniFL](https://arxiv.org/abs/2211.07300)

## Release Note
- 2023.09.16: Support MIMIC-IV 2.0 & Bug Fix

## How to Run
### Requirements
```
$ conda env create -n unifl -f env.yaml
```

### Preprocessing
- MIMIC-III, MIMIC-IV, eICU are public, but require some certificates
- Download the files from below links: 
    - [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
    - [eICU](https://physionet.org/content/eicu-crd/2.0/)
    - [MIMIC-IV](https://physionet.org/content/mimiciv/2.0/)

- Then, execute the preprocessing codes
```
$ cd preprocess
$ bash preprocess_run.sh {MIMIC-III} {MIMIC-IV} {eICU} {save_path}
```
- Note that the preprocessing takes about 1 hours with AMD EPYC 7502 32-core processor, and it requires more than 60GB of RAM.



### Model train
```
$ python src/main.py --device_num 0 --input_path <INPUT_PATH> --save_dir <SAVE_PATH> --train_type fedrated --algorithm fedpxn --pred_target mort --wandb_entity_name <ENTITY_NAME> --wandb_project_name <PROJECT_NAME>
```
or, you can execute multiple experiments simulatneously with `scheduler.py`

---
## NOTE
- Pause & Resume is only supported for fedrated learning (Kubernetis support)
- Pause & Resume is not verified with distributed environment
- We used one A100 80G gpu or two A6000 48G gpus for each run
- Distributed Data Parallel (DDP) with resume is not tested
- You can check hyperparameters on `main.py`

## Citation
```
@article{kim2022universal,
  title={Universal EHR federated learning framework},
  author={Kim, Junu and Hur, Kyunghoon and Yang, Seongjun and Choi, Edward},
  journal={arXiv preprint arXiv:2211.07300},
  year={2022}
}
```