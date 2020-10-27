# Learning Rolling Shutter Correction from Real Data
Copyright (c) <2020> <Jiawei Mo, Md Jahidul Islam, Junaed Sattar>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Usage
1. Dataset Process
  * Option a: Process data from raw TUM dataset
    * Download [TUM Rolling Shutter Dataset](https://vision.in.tum.de/data/datasets/rolling-shutter-dataset) with Euroc/DSO format, modify **data_path** in *tum_process/tum_process.py* accordingly
    * Download [PWC-Net weights](https://drive.google.com/file/d/1hB5nCbBJf6I06dL5VX4aiAdTYUzlLCsi/view?usp=sharing), modify **ckpt_path** in *tum_process/pwcnet.py* accordingly
    * Process the dataset (specify **save_path** in *tum_process/tum_process.py* if necessary)
      ```
      python3 tum_process/tum_process.py
      ```
  * Option b: Download the [processed data](https://drive.google.com/file/d/1AvMRv63N1czyJI2L2niFVN4Io-5Xlny6/view?usp=sharing) directly

2. Training (modify **data_path** in *data_loader.py* to the **save_path** or the **processed data**)
```
python3 -m train_depth
python3 -m train_anchor --anchor=1
python3 -m train_anchor --anchor=2
python3 -m train_anchor --anchor=4
...
```

3. Testing
```
python3 -m test  --anchor=1
python3 -m test  --anchor=2
python3 -m test  --anchor=4 --rectify_img=1
...
```

4. Plot errors
```
python3 -m view_errs
```

5. Check recitified images in */test_results/images*
