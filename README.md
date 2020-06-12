# regression-prior-networks
An official PyTorch implementation of "Regression Prior Networks" for effective uncertainty estimation.


Important! Whenever evaluatiing ensemble of Gaussian models, use the following:

```bash
python eval_nyu_model.py --checkpoint $(printf "./checkpoints/dense_depth_gaussian/%d/19.ckpt " {1..5})  --model_type "gaussian-ensemble" --targets_transform "scaled"
```
