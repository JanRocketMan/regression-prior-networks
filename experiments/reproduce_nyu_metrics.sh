printf "Single model 1 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_gaussian/1/19.ckpt"  --model_type "gaussian" --targets_transform "scaled" --calibration

printf "Single model 2 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_gaussian/2/19.ckpt"  --model_type "gaussian" --targets_transform "scaled" --calibration

printf "Single model 3 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_gaussian/3/19.ckpt"  --model_type "gaussian" --targets_transform "scaled" --calibration

printf "Single model 4 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_gaussian/4/19.ckpt"  --model_type "gaussian" --targets_transform "scaled" --calibration


printf "ENSM model test:\n"
python eval_nyu_model.py --checkpoint $(printf "./checkpoints/dense_depth_gaussian/%d/19.ckpt " {1..5})  --model_type "gaussian-ensemble" --targets_transform "scaled" --calibration


printf "EnD^2 model 1 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_endd/1/21.ckpt"  --model_type "nw_prior" --targets_transform "scaled" --calibration

printf "EnD^2 model 2 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_endd/2/21.ckpt"  --model_type "nw_prior" --targets_transform "scaled" --calibration

printf "EnD^2 model 3 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_endd/3/21.ckpt"  --model_type "nw_prior" --targets_transform "scaled" --calibration

printf "EnD^2 model 4 test:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_endd/4/21.ckpt"  --model_type "nw_prior" --targets_transform "scaled" --calibration
