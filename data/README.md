# data/

This folder holds the dataset locally. Its contents (except this file) are ignored by git: the raw data is not redistributed with this repository.

Dataset: "Electric Motor Temperature", Paderborn University, on Kaggle: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature (file `measures_v2.csv`).

The dataset licence has not been verified by the authors of this repository. Check the terms on the Kaggle page before any reuse.

Get the data:

```
python scripts/download_data.py
```

This uses `kagglehub` and needs Kaggle credentials (a `~/.kaggle/kaggle.json` file, or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables). You can also download `measures_v2.csv` by hand and place it here.

The code reads `data/measures_v2.csv` by default; set `MOTOR_DATA_PATH` to use another location.

Citation: Kirchgassner, W., Wallscheid, O., Bocker, J., "Data-Driven Permanent Magnet Temperature Estimation in Synchronous Motors With Supervised Machine Learning: A Benchmarking Study", IEEE Transactions on Power Electronics 36(7), 2021, doi 10.1109/tpel.2020.3045596.
