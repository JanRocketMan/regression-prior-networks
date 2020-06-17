printf "Single model:\n"
python ood_eval_nyu.py --checkpoint "checkpoints/dense_depth_gaussian/3/19.ckpt" \
    --model_type "gaussian" --measures "variance" "entropy" --targets_transform "scaled"

printf "ENSM model:\n"
python ood_eval_nyu.py --checkpoint $(printf "./checkpoints/dense_depth_gaussian/%d/19.ckpt " {1..5}) \
    --model_type "gaussian-ensemble" --measures "total_variance" "expected_pairwise_kl" "variance_of_expected" --targets_transform "scaled"

printf "EnD^2 model:\n"
python ood_eval_nyu.py --checkpoint "checkpoints/dense_depth_endd/4/21.ckpt" \
    --model_type "gaussian-ensemble" --measures "predicitve_posterior_entropy" "total_variance" \
        "mutual_information" "expected_pairwise_kl" "variance_of_expected" --targets_transform "scaled"
