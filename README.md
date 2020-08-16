# Learning Rolling Shutter Correction from Real Data
0. Download [TUM Rolling Shutter Dataset](https://vision.in.tum.de/data/datasets/rolling-shutter-dataset) with Euroc/DSO format
1. Process TUM Dataset
```
python3 tum_process/tum_process.py
```
2. Training
```
python3 -m train_depth
python3 -m train_anchor --num_anchor=1
python3 -m train_anchor --num_anchor=2
python3 -m train_anchor --num_anchor=3
python3 -m train_anchor --num_anchor=4
```
3. Testing
```
python3 -m test  --num_anchor=1
python3 -m test  --num_anchor=2
python3 -m test  --num_anchor=3
python3 -m test  --num_anchor=4
```
4. Check undistorted images in **_test_results_** folder, plot errors
```
python3 -m view_errs
```
