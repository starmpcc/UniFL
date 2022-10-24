#!bin/sh
rawdata_path=$1
inputdata_path=$2

python dataset_construct.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
python main_setp1.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
python main_step2.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
python npy_script.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
python split_for_FL.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;
python split_npy.py --rawdata_path $rawdata_path --inputdata_path $inputdata_path ;

