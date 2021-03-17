# MS2Planner 

## Input
It takes feature table generated from ```convert_to_table``` and raw signal .mzTab (curve mode) as input.

## Output
It outputs best paths in .txt format, path_finder.log is the log file.

## Run
```
python path_finder.py mode infile outfile intensity_threshold intensity_ratio num_path -infile_raw -intensity_accu -win_len -isolation -restriction -delay -min_scan -max_scan
```
Unit for retention time is second (sec) and mass-to-charge ratio is dalton (Da).

- ```mode```: ```apex``` or ```baseline``` or ```curve``` 
    - ```apex``` mode uses apex file and applies path finding algorithm to find best paths.
    - ```baseline``` mode simply splits retention time into small windows, length of the window is specified by ```win_len```. At each window, top ```num_path``` intensity ions are selected.
    - ```curve``` mode uses apex and raw MS1 file. It applies path finding algorithm to find best paths. 
- ```infile```: input file, feature tables generated from ```convert_to_table```. It contains five columns: mz, rt, charge, blank intensity, sample intensity. Blank and sample are told from the column index.
- ```outfile```: output file.
- ```intensity_threshold```: threshold for feature filtering. Any sample features below this threshold will be removed.
- ```intensity_ratio```: ratio threshold for feature filtering. Any sample features with sample_intensity / blank_intensity < ```intensity_ratio``` wil be removed.
- ```num_path```: number of paths.
- ```-infile_raw```: raw MS1 .mzML file (only include sample, no control), argument for ```curve``` mode. This file is generated from MS1mzTab.topppas. It contains three column: rt, mz and sample intensity.
- ```-intensity_accu```: the amount of intensity user wants to collect on a single feature, argument for ```apex``` and ```curve``` modes.
- ```-win_len```: (second) rt window length in baseline mode, argument for ```baseline``` mode. 
- ```-isolation```: (dalton) length of mass to charge isolation window, argument for all three modes.  
- ```-restriction```: (second, dalton) the first number is rt restriction and second is mz restriction. Restriction area is calculated as l1 norm. Features out of this area will not be involved, argument for ```curve``` mode. 
- ```-delay```: (second) the minimum rt requires swithching from one feature to the next, argument for all three modes.
- ```-min_scan```: (second) minimum scan period in acquistion, argument for ```apex``` and ```curve``` mode.
- ```-max_scan```: (second) maximum scan period in acquistion, argument for ```apex``` and ```curve``` mode.
- ```-cluster```: clustering algorithm for ```curve``` mode. kNN and GMM are provided.


## Output Format
```
path0 mz_center1 mz_isolation1 duration1 rt_start1 rt_end1 intensity1 apex_rt1 charge1 \t mz_center2 mz_center2 mz_window2 duration2 rt_start2 rt_end2 intensity2 apex_rt2 charge2...
path1 mz_center1 mz_isolation1 duration1 rt_start1 rt_end1 intensity1 apex_rt1 charge1...
```
Each path contains a row in .txt file. Following by
- ```mz_center```: the center of the mz_window
- ```mz_isolation```: the length of the half of mz_window (i.e. true sampling window should be (mz_center - mz_isolation: mz_center + mz_isolation))
- ```duration```: length of collecting (```rt_end``` - ```rt_start```) 
- ```rt_start```: start of rt
- ```rt_end```: end of rt
- ```intensity```: intensity of the apex feature 
- ```apex_rt```: the retention time of the apex feature
- ```charge```: charge of the apex feature

These numbers are separated by **space**. Different sampling position are separated by **\t**. (i.e. ```rt_end1``` and ```mz_center2``` are separated by **\t**).

## Test
- Generate .csv from ```convert_to_table```
- Unzip the test data in ./test folder
- Run command line, parameters used for test are as follows
### Apex mode
```
python path_finder.py apex test/SA113_Media_SPE_MeOH_MS1_to_SA113_SPE_MeOH_MS1_table.csv test/path_5_apex.txt 1e5 3 5 -intensity_accu 1e5 -isolation 1 -delay 0.2 -min_scan 0.2 -max_scan 3
```
### Baseline Mode
```
python path_finder.py baseline test/SA113_Media_SPE_MeOH_MS1_to_SA113_SPE_MeOH_MS1_table.csv test/path_5_baseline.txt 1e5 3 5 -win_len 0.5 -isolation 1 -delay 0.2
```

### Curve Mode
```
python path_finder.py curve test/SA113_Media_SPE_MeOH_MS1_to_SA113_SPE_MeOH_MS1_table.csv test/path_5_curve.txt 1e5 3 5 -infile_raw test/SA113_SPE_MeOH_MS1.mzTab -intensity_accu 1e5 -restriction 2 0.2 -isolation 1 -delay 0.2 -min_scan 0.2 -max_scan 3 -cluster kNN
```

## TOPPAS Run
- Raw MS1 data to .mzTab file (for curve method input)
```
ExecutePipeline.exe -in MS1mzTab.toppas -out_dir ./total_ion_curr/data/MS1
```