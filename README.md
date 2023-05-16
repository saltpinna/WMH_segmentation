# Automatic segmentation of White Matter Hyperintensities in brain T1-weighted and FLAIR MRI images, classification according to Fazekas scale and spatial and T1-signal intensity based subdivision

Instructions for running the WMH segmentation pipeline.

By: Inna Fryckstedt 

## Prerequisites

To be able to run the pipeline, nnUNet (v1) must be installed on your GPU machine. Follow the installation instructions here: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1. 


Download the nnUNet_folders directory, which can be found in https://drive.google.com/drive/folders/1_oczQyPC7oZx2jQSirIGrlXxkdpbjA7K?usp=sharing. Set the environment variables to the location of the provided nnUNet_folders directory:

export nnUNet_raw_data_base="/path/to/nnUNet_folders/nnUNet_raw_data_base"

export nnUNet_preprocessed="/path/to/nnUNet_folders/nnUNet_preprocessed"

export RESULTS_FOLDER="/path/to/nnUNet_folders/nnUNet_trained_models"


In the nnU-Net source code, add the provided files: nnUNetTrainerV2_150epochs.py & nnUNetTrainerV2_200epochs.py (found in nnUNet_trainers) to the nnUNet/nnunet/training/network_training/ directory. 


## Running the pipeline
In the WMH_segmentation.py file, provide the paths to the T1 and FLAIR images and output directory for the results and run the script. The two input images should be located in a directory named as the case identifier. The output directory must contain a FreeSurfer license text file named fs_license.txt, which can be found here: https://surfer.nmr.mgh.harvard.edu/fswiki/License. 

By default, the pipeline uses the continuity to ventricles (connected components) method for spatial subdivision of WMHs, the 'spatial_subdivision' parameter can be set to either 'connected_components', '10_mm_dilation', or 'four_categories' when calling the WMH_segmentation function, depending on the spatial subdivision method of choice. 


FastSurfer, which is used for tissue segmentation, is run through a Docker container which requires root privileges. During the first run of this pipeline, you will be prompted to enter the sudo password for the current user during the process. 


The final WMH segmentation image will be located in the ensemble_preds directory in the output path and the subdivision based on spatial location and T1-signal intensity in t1_subdivision in the case identifier directory (intensity_subdivision.nii.gz).

