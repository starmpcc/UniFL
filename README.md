# Universal EHR Federated Learning Framework
This repository is the codes for [UniFL](https://arxiv.org/abs/2211.07300)
## How to Run
### Requirements
```
$ conda create env -f env.yaml
```

### Preprocessing
- MIMIC-III, MIMIC-IV, eICU are public, but require some certificates
- Download the files from below links: 
    - [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
    - [eICU](https://physionet.org/content/eicu-crd/2.0/)
    - [MIMIC-IV](https://physionet.org/content/mimiciv/0.3/)
    - [2018_I10gem.txt](https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-CM-General-Equivalence-Mappings.zip)
    - [ccs_multi_dx_tool_2015](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip)
    - [icd10cmtoicd9gem](https://data.nber.org/gem/icd10cmtoicd9gem.csv)

- Move the files like below structure
```
<rawdata_path>
├─ mimic3
│  ├─ ADMISSIONS.csv
│  ├─ PATIENTS.csv
│  ├─ ICUSYAYS.csv
│  ├─ LABEVENTES.csv
│  ├─ PRESCRIPTIONS.csv
│  ├─ PROCEDURES.csv
│  ├─ INPUTEVENTS_CV.csv
│  ├─ INPUTEVENTS_MV.csv
│  ├─ D_ITEMDS.csv
│  ├─ D_ICD_PROCEDURES.csv
│  └─ D_LABITEMBS.csv
├─ eicu
│  ├─ diagnosis.csv
│  ├─ infusionDrug.csv
│  ├─ lab.csv
│  ├─ medication.csv
│  └─ patient.csv
├─ mimic4
│  ├─ admissions.csv
│  ├─ ...
│  └─ 2018_I10gem.txt
├─ ccs_multi_dx_tool_2015.csv
└─ icd10cmtoicd9gem.csv

```

- Then, execute the preprocessing codes
```
$ bash preprocess_run.sh <rawdata_path> <inputdata_path>
```
- Note that the preprocessing takes about 1 hours with AMD EPYC 7502 32-core processor, and it requires more than 60GB of RAM.

- If you want to preprocess in step by step way, command each preprocess code by following below,
```
$ python dataset_construct.py --raw_path <rawdata_path> --input_path <inputdata_path>
$ python main_step1.py --raw_path <rawdata_path> --input_path <inputdata_path>
$ python main_step2.py --raw_path <rawdata_path> --input_path <inputdata_path>
$ python npy_script.py --raw_path <rawdata_path> --input_path <inputdata_path>
$ python split_for_FL.py --raw_path <rawdata_path> --input_path <inputdata_path>
$ python split_npy.py --raw_path <rawdata_path> --input_path <inputdata_path>
```


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