bash ./scripts/dist_train.sh 4 --cfg_file ./cfgs/omega_models/pv_rcnn.yaml --extra_tag 20e_020_4x4
bash ./scripts/dist_train.sh 4 --cfg_file ./cfgs/omega_models/pv_rcnn.yaml --extra_tag 40e_020_4x4 --epochs 40
bash ./scripts/dist_train.sh 4 --cfg_file ./cfgs/omega_models/cbgs_pv_rcnn.yaml --extra_tag 40e_020_4x4 --epochs 40
