printf "Single model:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_gaussian/3/19.ckpt"  --model_type "gaussian" --targets_transform "scaled"

printf "ENSM model:\n"
python eval_nyu_model.py --checkpoint $(printf "./checkpoints/dense_depth_gaussian/%d/19.ckpt " {1..5})  --model_type "gaussian-ensemble" --targets_transform "scaled"

printf "EnD^2 model:\n"
python eval_nyu_model.py --checkpoint "checkpoints/dense_depth_endd/4/21.ckpt"  --model_type "nw_prior" --targets_transform "scaled"
