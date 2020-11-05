# Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption
Copyright (C) <2020> <Jiawei Mo, Md Jahidul Islam, Junaed Sattar>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
