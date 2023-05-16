# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:08:37 2023 by
Dimitris Toumpanakis, MD
M.Sc. in Data Science and Machine Learning
PhD fellow
"""
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
import os

def create_distance_map(pred_data_path, output_folder):
    # Brain structures dictionary
    roi = {
    'Left-Cortical-White-Matter': 2,
    'Left-Lateral-Ventricle': 4,
    'Left-Inf-Lat-Vent': 5,
    'Left-Cerebellum-White-Matter': 7,
    'Left-Cerebellum-Cortex': 8,
    'Left-Thalamus-Proper': 10,
    'Left-Caudate': 11,
    'Left-Putamen': 12,
    'Left-Pallidum': 13,
    'Left-3rd-Ventricle': 14,
    'Left-4th-Ventricle': 15,
    'Left-Brain-Stem': 16,
    'Left-Hippocampus': 17,
    'Left-Amygdala': 18,
    'Left-CSF': 24,
    'Left-Accumbens-area': 26,
    'Left-VentralDC': 28,
    'Left-choroid-plexus': 31,
    'Right-Cortical-White-Matter': 41,
    'Right-Lateral-Ventricle': 43,
    'Right-Inf-Lat-Vent': 44,
    'Right-Cerebellum-White-Matter': 46,
    'Right-Cerebellum-Cortex': 47,
    'Right-Thalamus-Proper': 49,
    'Right-Caudate': 50,
    'Right-Putamen': 51,
    'Right-Pallidum': 52,
    'Right-Hippocampus': 53,
    'Right-Amygdala': 54,
    'Right-Accumbens-area': 58,
    'Right-VentralDC': 60,
    'Right-choroid-plexus': 63,
    'Right-3rd-Ventricle': 14,
    'Right-4th-Ventricle': 15,
    'Right-Brain-Stem': 16,
    'Right-CSF': 24,
    'ctx-lh-caudalanteriorcingulate': 1002,
    'ctx-lh-caudalmiddlefrontal': 1003,
    'ctx-lh-cuneus': 1005,
    'ctx-lh-entorhinal': 1006,
    'ctx-lh-fusiform': 1007,
    'ctx-lh-inferiorparietal': 1008,
    'ctx-lh-inferiortemporal': 1009,
    'ctx-lh-isthmuscingulate': 1010,
    'ctx-lh-lateraloccipital': 1011,
    'ctx-lh-lateralorbitofrontal': 1012,
    'ctx-lh-lingual': 1013,
    'ctx-lh-medialorbitofrontal': 1014,
    'ctx-lh-middletemporal': 1015,
    'ctx-lh-parahippocampal': 1016,
    'ctx-lh-paracentral': 1017,
    'ctx-lh-parsopercularis': 1018,
    'ctx-lh-parsorbitalis': 1019,
    'ctx-lh-parstriangularis': 1020,
    'ctx-lh-pericalcarine': 1021,
    'ctx-lh-postcentral': 1022,
    'ctx-lh-posteriorcingulate': 1023,
    'ctx-lh-precentral': 1024,
    'ctx-lh-precuneus': 1025,
    'ctx-lh-rostralanteriorcingulate': 1026,
    'ctx-lh-rostralmiddlefrontal': 1027,
    'ctx-lh-superiorfrontal': 1028,
    'ctx-lh-superiorparietal': 1029,
    'ctx-lh-superiortemporal': 1030,
    'ctx-lh-supramarginal': 1031,
    'ctx-lh-transversetemporal': 1034,
    'ctx-lh-insula': 1035,
    'ctx-rh-caudalanteriorcingulate': 2002,
    'ctx-rh-caudalmiddlefrontal': 2003,
    'ctx-rh-cuneus': 2005,
    'ctx-rh-entorhinal': 2006,
    'ctx-rh-fusiform': 2007,
    'ctx-rh-inferiorparietal': 2008,
    'ctx-rh-inferiortemporal': 2009,
    'ctx-rh-isthmuscingulate': 2010,
    'ctx-rh-lateraloccipital': 2011,
    'ctx-rh-lateralorbitofrontal': 2012,
    'ctx-rh-lingual': 2013,
    'ctx-rh-medialorbitofrontal': 2014,
    'ctx-rh-middletemporal': 2015,
    'ctx-rh-parahippocampal': 2016,
    'ctx-rh-paracentral': 2017,
    'ctx-rh-parsopercularis': 2018,
    'ctx-rh-parsorbitalis': 2019,
    'ctx-rh-parstriangularis': 2020,
    'ctx-rh-pericalcarine': 2021,
    'ctx-rh-postcentral': 2022,
    'ctx-rh-posteriorcingulate': 2023,
    'ctx-rh-precentral': 2024,
    'ctx-rh-precuneus': 2025,
    'ctx-rh-rostralanteriorcingulate': 2026,
    'ctx-rh-rostralmiddlefrontal': 2027,
    'ctx-rh-superiorfrontal': 2028,
    'ctx-rh-superiorparietal': 2029,
    'ctx-rh-superiortemporal': 2030,
    'ctx-rh-supramarginal': 2031,
    'ctx-rh-transversetemporal': 2034,
    'ctx-rh-insula': 2035 
    }

    # Load the segmentation output of fastsurfer ("aparc.DKTatlas+aseg.deep.nii.gz") 
    pred_data = nib.load(pred_data_path).get_fdata()
    pred = nib.load(pred_data_path)
    affine = pred.affine
    spacings = pred.header.get_zooms()

    # Input the label numbers for ventricles
    # (in that case the lateral ventricles, i.e. labels 4,5,43,44)
    ventricle_label_nums = [4,5,43,44]

    # Create a boolean mask for the region of interest
    mask = np.zeros_like(pred_data)
    for label_num in ventricle_label_nums:
        mask = np.logical_or(mask, pred_data == label_num)

    # Convert the boolean masks to integer masks
    mask = mask.astype(np.int) * 100 
    mask_map = np.logical_not(mask) # ~ creates inverted logic (True -> False), to create the distance map

    # Create distance map to ventricles
    mask_dist_map = distance_transform_edt(mask_map, sampling=spacings)

    # Create a Nifti image from the masks
    mask_dist_map_img = nib.Nifti1Image(mask_dist_map, affine=affine)
    nib.save(mask_dist_map_img, output_folder + '/' + pred_data_path.split('/')[-3] + '_distance_map_ventricles.nii.gz')

    # Outer border/contour of the cortical white matter, labels 2 and 41
    label_nums = [2,41]

    # Create a boolean mask for the region of interest
    mask = np.zeros_like(pred_data)
    for label_num in label_nums:
        mask = np.logical_or(mask, pred_data == label_num)

    # Convert the boolean masks to integer masks
    mask = mask.astype(np.int) * 100 
    mask_map = np.logical_not(mask) # ~ creates inverted logic (True -> False), to create the distance map

    # Create distance map to ventricles
    mask_dist_map = distance_transform_edt(mask_map, sampling=spacings)

    # Create a Nifti image from the masks
    mask_dist_map_img = nib.Nifti1Image(mask_dist_map, affine=affine)
    nib.save(mask_dist_map_img, output_folder + '/' + pred_data_path.split('/')[-3] + '_distance_map_cort_junction.nii.gz')

if __name__ == '__main__':
    cases = os.listdir('/home/neuro_nph/Desktop/WMH_segmentation_2/fastsurfer/my_fastsurfer_analysis/')
    for case in cases[:1]:
        case = case.split('_')[0]
        print(case)
        pred_data_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/fastsurfer/my_fastsurfer_analysis/' + case + '_0000.nii.gz/mri/aparc.DKTatlas+aseg.deep.mgz'
        output_folder = '/home/neuro_nph/Desktop/WMH_segmentation_2/fastsurfer/my_fastsurfer_analysis/' + case + '_0000.nii.gz'
        create_distance_map(pred_data_path, output_folder)