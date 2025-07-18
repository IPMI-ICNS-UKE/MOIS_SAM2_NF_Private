import os
from pathlib import Path
import shutil
from tqdm import tqdm


# Define mapping based on the image content
testset_mappings = {
    "TestSet_1": {
        "nf_000_0000.nii.gz": "2_2013_P_502_T2.nii.gz",
        "nf_001_0000.nii.gz": "5_2018_P_294_T2.nii.gz",
        "nf_002_0000.nii.gz": "5_2018_P_523_T2.nii.gz",
        "nf_003_0000.nii.gz": "6_2018_P_271_T2.nii.gz",
        "nf_004_0000.nii.gz": "6_2018_P_469_T2.nii.gz",
        "nf_005_0000.nii.gz": "6_2019_P_109_T2.nii.gz",
        "nf_006_0000.nii.gz": "6_2020_P_473_T2.nii.gz",
        "nf_007_0000.nii.gz": "7_2014_P_82_T2.nii.gz",
        "nf_008_0000.nii.gz": "7_2018_P_304_T2.nii.gz",
        "nf_009_0000.nii.gz": "7_2020_P_678_T2.nii.gz",
        "nf_010_0000.nii.gz": "8_2018_P_197_T2.nii.gz",
        "nf_011_0000.nii.gz": "9_2019_P_377_T2.nii.gz",
        "nf_012_0000.nii.gz": "9_2020_P_604_T2.nii.gz",
    },
    # "TestSet_2": {
    #     "nf_000.nii.gz": "1_2006_P_637_T2.nii.gz",
    #     "nf_001.nii.gz": "1_2007_P_26_T2.nii.gz",
    #     "nf_002.nii.gz": "1_2007_P_531_T2.nii.gz",
    #     "nf_003.nii.gz": "1_2008_P_291_T2.nii.gz",
    #     "nf_004.nii.gz": "1_2008_P_316_T2.nii.gz",
    #     "nf_005.nii.gz": "1_2008_P_564_T2.nii.gz",
    #     "nf_006.nii.gz": "1_2009_P_722_T2.nii.gz",
    #     "nf_007.nii.gz": "2_2009_P_637_T2.nii.gz",
    #     "nf_008.nii.gz": "2_2011_P_26_T2.nii.gz",
    #     "nf_009.nii.gz": "4_2012_P_701_T2.nii.gz",
    #     "nf_010.nii.gz": "5_2012_P_430_T2.nii.gz",
    # },
    # "TestSet_3": {
    #     "nf_000.nii.gz": "1_2013_P_852_T2.nii.gz",
    #     "nf_001.nii.gz": "1_2021_P_212_T2.nii.gz",
    #     "nf_002.nii.gz": "2_2013_P_303_T2.nii.gz",
    #     "nf_003.nii.gz": "2_2014_P_110_T2.nii.gz",
    #     "nf_004.nii.gz": "2_2014_P_851_T2.nii.gz",
    #     "nf_005.nii.gz": "2_2014_P_852_T2.nii.gz",
    #     "nf_006.nii.gz": "2_2016_P_509_T2.nii.gz",
    #     "nf_007.nii.gz": "3_2013_P_56_T2.nii.gz",
    #     "nf_008.nii.gz": "3_2014_P_319_T2.nii.gz",
    #     "nf_009.nii.gz": "3_2016_P_274_T2.nii.gz",
    #     "nf_010.nii.gz": "3_2016_P_851_T2.nii.gz",
    #     "nf_011.nii.gz": "3_2017_P_303_T2.nii.gz",
    #     "nf_012.nii.gz": "3_2017_P_764_T2.nii.gz",
    #     "nf_013.nii.gz": "3_2019_P_110_T2.nii.gz",
    #     "nf_014.nii.gz": "3_2019_P_852_T2.nii.gz",
    #     "nf_015.nii.gz": "4_2016_P_56_T2.nii.gz",
    #     "nf_016.nii.gz": "4_2018_P_319_T2.nii.gz",
    #     "nf_017.nii.gz": "4_2019_P_274_T2.nii.gz",
    #     "nf_018.nii.gz": "4_2020_P_303_T2.nii.gz",
    #     "nf_019.nii.gz": "5_2018_P_56_T2.nii.gz",
    #     "nf_020.nii.gz": "6_2020_P_319_T2.nii.gz",
    #     "nf_021.nii.gz": "6_2020_P_56_T2.nii.gz",
    # },
    # "TestSet_4": {
    #     "nf_001.nii.gz": "Patient_1_T2_STIR.nii.gz",
    #     "nf_002.nii.gz": "Patient_2_T2_STIR.nii.gz",
    #     "nf_003.nii.gz": "Patient_3_T2_STIR.nii.gz",
    #     "nf_004.nii.gz": "Patient_4_T2_STIR.nii.gz",
    #     "nf_005.nii.gz": "Patient_5_T2_STIR.nii.gz",
    #     "nf_006.nii.gz": "Patient_6_T2_STIR.nii.gz",
    #     "nf_007.nii.gz": "Patient_7_T2_STIR.nii.gz",
    #     "nf_008.nii.gz": "Patient_8_T2_STIR.nii.gz",
    #     "nf_009.nii.gz": "Patient_12_T2_STIR.nii.gz",
    #     "nf_010.nii.gz": "Patient_15_T2_STIR.nii.gz",
    #     "nf_011.nii.gz": "Patient_21_T2_STIR.nii.gz",
    # },
}

# Base directory to apply renaming (to be adapted)
base_dir_list = [Path("/home/gkolokolnikov/PhD_project/nf_segmentation_interactive/NFInteractiveSegmentationBenchmarkingPrivate/evaluation/results/predictions/unet/global_non_corrective"),
                 ]

# Function to perform renaming
def rename_files(base_dir, mappings):
    for testset, file_map in mappings.items():
        for fold in tqdm(range(1, 4)):
            fold_path = base_dir / testset / f"fold_{fold}"
            if not fold_path.exists():
                continue
            for old_name, new_name in file_map.items():
                old_path = fold_path / old_name
                new_path = fold_path / new_name
                if old_path.exists():
                    shutil.move(str(old_path), str(new_path))

for base_dir in base_dir_list:
    print("Processing: ", base_dir)
    rename_files(base_dir, testset_mappings)
