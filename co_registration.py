import ants
import nibabel as nib
import os

path = r"C:/Users/46722/Documents/Exjobb/scripts/WMH_segmentation/data"
fixed_img_path = nib.load(path + '/training/Amsterdam/GE3T/100/orig/FLAIR.nii.gz')
moved_img_path = nib.load(path + '/training/Amsterdam/GE3T/100/orig/3DT1.nii.gz')

fixed_img = fixed_img_path.get_fdata()
moved_img = moved_img_path.get_fdata()

# fixed_img = ants.resample_image(fixed_img, (256, 256, 176), 1, 0)
flat_img = ants.get_ants_data(fixed_img)
print(fixed_img.shape, moved_img.shape)