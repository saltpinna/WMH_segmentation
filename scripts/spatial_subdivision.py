import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from scipy.ndimage import label
import os

def WMH_connected_components(wmh_mask_path, dist_map_path, output_path):
    """
    Creates a new label image, where 0:background, 1:DWMH & 2:PWMH.
    The method classifies continued WMH clusters touching ventricle walls as PWMHs, and the rest as DWMHs.
    """

    # Load your images
    dist_map_im = nib.load(dist_map_path)
    wmh_mask_im = nib.load(wmh_mask_path)

    # Get the affine and target shape of the `wmh_mask_im` image
    target_affine = wmh_mask_im.affine
    target_shape = wmh_mask_im.shape

    # Resample your `dist_map_im` image to the target shape and affine
    dist_map_im_resized = resample_img(dist_map_im, target_affine=target_affine, target_shape=target_shape)

    # Turn images into arrays
    wmh_mask = wmh_mask_im.get_fdata()
    dist_map = dist_map_im_resized.get_fdata()

    if wmh_mask.shape != dist_map.shape:
        print('Dimension mismatch of inputs')
        print(wmh_mask.shape)
        print(dist_map.shape)
    else:
        wmh_mask[wmh_mask > 0] = 1
        L, num_features = label(wmh_mask, structure=np.ones((3,3,3)))
        for j in range(1, num_features+1):
            if np.min(dist_map[L == j]) <= 2:
                wmh_mask[L == j] = 2
            else:
                wmh_mask[L == j] = 1
        connected_components_mask = nib.Nifti1Image(wmh_mask, wmh_mask_im.affine)
        nib.save(connected_components_mask, output_path + '/' + wmh_mask_path.split('/')[-1].split('WMH_')[-1].split('.nii.gz')[0] + '_wmh_connected_components.nii.gz')


def WMH_dilation_mask_10mm(wmh_mask_path, dist_map_path, output_path):
    """
    Creates a new label image, where 0:background, 1:DWMH & 2:PWMH.
    The method classifies WMH voxels within 10 mm from the ventricle wall as PWMHs and the rest as DWMHs.
    """

    # Load your images
    dist_map_im = nib.load(dist_map_path)
    wmh_mask_im = nib.load(wmh_mask_path)

    # Get the affine and target shape of the `wmh_mask_im` image
    target_affine = wmh_mask_im.affine
    target_shape = wmh_mask_im.shape

    # Resample your `dist_map_im` image to the target shape and affine
    dist_map_im_resized = resample_img(dist_map_im, target_affine=target_affine, target_shape=target_shape)

    wmh_mask = wmh_mask_im.get_fdata()
    dist_map = dist_map_im_resized.get_fdata()

    if wmh_mask.shape != dist_map.shape:
        print('Dimension mismatch of inputs')
        print(wmh_mask.shape)
        print(dist_map.shape)
    else:
        wmh_mask[wmh_mask > 0] = 1
        wmh_mask[(dist_map >= 0) & (dist_map <= 10) & (wmh_mask == 1)] = 2
        dilated_mask = nib.Nifti1Image(wmh_mask, wmh_mask_im.affine)
        nib.save(dilated_mask, output_path + '/' + wmh_mask_path.split('/')[-1].split('WMH_')[-1].split('.nii.gz')[0] + '_wmh_10_mm_dilation.nii.gz')


def WMH_four_categories(wmh_mask_path, dist_map_path, cort_map_path, output_path):
    """
    Creates a new label image, where 0:background, 1:DWMH, 2:PWMH, 3:JVWMH & 4:JCWMH.
    The method classifies WMH voxels within 3 mm from the ventricle wall as JVWMHs, between 3mm-13mm as PWMHs, 
    within 4 mm of the corticomedullary junction as JCWMHs and the rest as DWMHs.
    """

    # Load your images
    dist_map_im = nib.load(dist_map_path)
    wmh_mask_im = nib.load(wmh_mask_path)
    cort_dist_map_im = nib.load(cort_map_path)

    # Get the affine and target shape of the `wmh_mask_im` image
    target_affine = wmh_mask_im.affine
    target_shape = wmh_mask_im.shape

    # Resample your `dist_map_im` image to the target shape and affine
    dist_map_im_resized = resample_img(dist_map_im, target_affine=target_affine, target_shape=target_shape)
    cort_dist_map_im_resized = resample_img(cort_dist_map_im, target_affine=target_affine, target_shape=target_shape)
   
    wmh_mask = wmh_mask_im.get_fdata()
    dist_map = dist_map_im_resized.get_fdata()
    cort_dist_map = cort_dist_map_im_resized.get_fdata()

    if (wmh_mask.shape != dist_map.shape) or (wmh_mask.shape != cort_dist_map.shape) or (dist_map.shape != cort_dist_map.shape):
        print('Dimension mismatch of inputs')
        print(wmh_mask.shape)
        print(dist_map.shape)
        print(cort_dist_map.shape)
    else:
        wmh_mask[wmh_mask > 0] = 1
        wmh_mask[(dist_map >= 0) & (dist_map <= 13) & (wmh_mask == 1)] = 2
        wmh_mask[(dist_map >= 0) & (dist_map <= 3) & (wmh_mask == 2)] = 3
        wmh_mask[(cort_dist_map <= 4) & (wmh_mask == 1)] = 4
        dilated_mask = nib.Nifti1Image(wmh_mask, wmh_mask_im.affine)
        nib.save(dilated_mask, output_path + '/' + wmh_mask_path.split('/')[-1].split('WMH_')[-1].split('.nii.gz')[0] + '_wmh_four_categories.nii.gz')