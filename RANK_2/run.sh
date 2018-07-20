python utils.py -set_params N 1
python _1_encode_cat_features.py
python _2b_generate_dataset_for_vw_fm.py
python _2c_generate_fm_features.py
python _3a_rf.py
python _3b_gbdt.py
python _3c_vw.py -rseed 1
python _3c_vw.py -rseed 2
python _3c_vw.py -rseed 3
python _3c_vw.py -rseed 4
python _3d_fm.py -rseed 51
python _3d_fm.py -rseed 52
python _3d_fm.py -rseed 53
python _3d_fm.py -rseed 54
#should generate logloss ~= 0.3937
python _4_post_processing.py

exit




