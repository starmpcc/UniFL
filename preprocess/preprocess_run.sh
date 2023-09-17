#!bin/sh
mimic3_path=$1
mimic4_path=$2
eicu_path=$3
save_path=$4

python dataset_construct.py --mimic3_path $mimic3_path --mimic4_path $mimic4_path --eicu_path $eicu_path --save_path $save_path ;
python main_step1.py --mimic3_path $mimic3_path --mimic4_path $mimic4_path --eicu_path $eicu_path --save_path $save_path ;
python main_step2.py --save_path $save_path ;
python split_for_FL.py --save_path $save_path --mimic3_path $mimic3_path --eicu_path $eicu_path ;
python split_npy.py --save_path $save_path ;