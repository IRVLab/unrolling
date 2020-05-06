# Learning Rolling Shutter Correction from Real Data
0. Download [TUM Rolling Shutter Dataset](https://vision.in.tum.de/data/datasets/rolling-shutter-dataset) with Euroc/DSO format
1. Process TUM Dataset
```
python3 tum_process/tum_process.py
```
2. Training
```
python3 -m unrollnet.train
python3 -m evaluation.depthnet.train
python3 -m evaluation.velocitynet.train
```
3. Testing
```
python3 -m evaluation.test
```
4. Check undistorted images in **_test_results_** folder, plot errors
```
python3 -m evaluation.view_errs
```
