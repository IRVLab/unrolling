# unrolling
1. Stereo rectify
```
python3 tum_rectify.py
```
2. Get disparity by VI-DSO
```
./VI-Stereo-DSO/run.bash
```
3. Get optical flow by PWC-Net
```
python3 pwcnet_get_flow.py
```
4. Training
```
python3 hfnet_train.py
```
5. Testing
```
python3 hfnet_test.py
```