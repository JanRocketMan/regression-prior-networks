# Regression Prior Networks
An official PyTorch implementation of "Regression Prior Networks" for effective uncertainty estimation.

# Results
Example
![result1](plots/example_nyu_630.png)

Performance metrics
![result2](plots/performance-metrics.png)

Calibration of predicted deviations vs ground truth errors
![result3](plots/mean-calibration.png)

# Installation
This repo was tested on Python 3.7.6 and PyTorch 1.4.0

All other requirements can be installed with conda
```bash
conda env create -f requirements.yml
```

# Data & Models

For Nyu training, we use the subsampled data (50K) from the DenseDepth repo:

[Train data](https://tinyurl.com/nyu-data-zip) (4.1 GB) and [test data](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip) (1.4 GB). 

store all zip files in data folder without unpacking them.

All trained checkpoints (ensemble of gaussians, our model) can be found [here](https://drive.google.com/drive/folders/1jL-g3yALvhilkNFlP6_ahtb7PY6XqxIW?usp=sharing) (1.6 GB).

Those should be extracted in checkpoints folder.

# Usage

To reproduce reported test metrics (table 3), run

```bash
bash experiments/reproduce_nyu_metrics.sh
```

# ToDo

- [x] Advanced visualization of results
- [ ] Training script
- [ ] Evaluation on KITTI

# Reference

@article{Malinin20,
  author    = {Andrey Malinin, Sergey Chervontsev, Ivan Provilkov, Mark Gales},
  title     = {Regression Prior Networks},
  journal   = {arXiv e-prints},
  volume    = {abs/TOFILL},
  year      = {2018},
  url       = {TOFILL},
  eid       = {TOFILL},
  eprint    = {TOFILL}
}
