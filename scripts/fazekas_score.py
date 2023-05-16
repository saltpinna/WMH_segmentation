import nibabel as nib
import numpy as np

def fazekas_score(wmh_mask_path, brain_mask_path):
    wmh_mask = nib.load(wmh_mask_path)
    brain_mask = nib.load(brain_mask_path)

    voxel_size_wmh = wmh_mask.header['pixdim'][1:4]
    voxel_vol_wmh = voxel_size_wmh[0] * voxel_size_wmh[1] * voxel_size_wmh[2]
    voxel_size_brain = brain_mask.header['pixdim'][1:4]
    voxel_vol_brain = voxel_size_brain[0] * voxel_size_brain[1] * voxel_size_brain[2]

    wmh_vol = np.count_nonzero(wmh_mask.get_fdata()) * voxel_vol_wmh / 1000  #Volume in ml
    brain_vol = np.count_nonzero(brain_mask.get_fdata()) * voxel_vol_wmh / 1000  #Volume in ml
    wmh_vol_normalized = wmh_vol / brain_vol * 100  # WMH volume in %

    if wmh_vol_normalized < 0.7:
        fazekas = 0
    elif (wmh_vol_normalized >= 0.7) and (wmh_vol_normalized < 2.5):
        fazekas = 1
    elif (wmh_vol_normalized >= 2.5) and (wmh_vol_normalized < 4.6):
        fazekas = 2
    elif wmh_vol_normalized >= 4.6:
        fazekas = 3

    return fazekas

if __name__ == '__main__':
    test_case = '75'
    wmh_mask_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/data/preprocessed_files/test/3DT1_FLAIR/test_ensemble_masks/WMH_' + test_case + '.nii.gz'
    brain_mask_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/data/preprocessed_files/test/3DT1_FLAIR/test_skullstripping_masks/' + test_case + '.nii.gz'    
    print('Fazekas score:', fazekas_score(wmh_mask_path, brain_mask_path))