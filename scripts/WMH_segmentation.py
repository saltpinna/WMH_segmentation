import subprocess
import os
from preprocessing import preprocessing
from spatial_subdivision import *
from create_distance_map import create_distance_map
from intensity_subdivision import intensity_subdivision
from fazekas_score import fazekas_score
import time

def WMH_segmentation(t1_image_path, flair_image_path, output_path, spatial_subdivision='connected_components'):
    
    # Preprocessing images
    print('Pre-processing images')
    preprocessing(t1_image_path, flair_image_path, output_path)
    
    # Running prediction on 2d nnU-Net
    print('Runnning inference on 2d nnU-Net')
    if not os.path.isdir(output_path + '/2d_preds/'):
        os.mkdir(output_path + '/2d_preds/')
    command = 'nnUNet_predict -i ' + output_path + '/final_preprocessed_images \
            -o ' + output_path + '/2d_preds' + ' -tr nnUNetTrainerV2_150epochs -ctr nnUNetTrainerV2CascadeFullRes \
            -m 2d -p nnUNetPlansv2.1 -t Task501_WMH --save_npz'
    subprocess.run(command, shell=True)

    # Running prediction on 3d nnU-Net
    print('Running inference on 3d nnU-net')
    if not os.path.isdir(output_path + '/3d_preds/'):
        os.mkdir(output_path + '/3d_preds/')
    command = 'nnUNet_predict -i ' + output_path + '/final_preprocessed_images \
            -o ' + output_path + '/3d_preds' + ' -tr nnUNetTrainerV2_200epochs -ctr nnUNetTrainerV2CascadeFullRes \
            -m 3d_fullres -p nnUNetPlansv2.1 -t Task501_WMH --save_npz'
    subprocess.run(command, shell=True)
    
    # Running ensemble
    print('Running ensemble')
    if not os.path.isdir(output_path + '/ensemble_preds/'):
        os.mkdir(output_path + '/ensemble_preds/')
    command = 'nnUNet_ensemble -f ' + output_path + '/2d_preds ' + output_path + '/3d_preds' + ' -o ' + output_path + 'ensemble_preds'
    subprocess.run(command, shell=True)

    # Calculating Fazekas score
    print('Calculating Fazekas score')
    score = fazekas_score(output_path + '/ensemble_preds/WMH_' + t1_image_path.split('/')[-2] + '.nii.gz', output_path + '/brain_masks/' + t1_image_path.split('/')[-2] + '.nii.gz')
    with open(output_path + '/fazekas_score.txt', 'w') as f:
        f.write(f"Fazekas score: {score}")
    
    # Running fastsurfer for tissue segmentation
    print('Running FastSurfer for tissue segmentation')
    if not os.path.isdir(output_path + '/fastsurfer/'):
        os.mkdir(output_path + '/fastsurfer/')
    command = 'sudo docker pull deepmi/fastsurfer'
    subprocess.run(command, shell=True)
    command = 'sudo docker run --gpus all -v ' + output_path + '/final_preprocessed_images:/data -v ' + output_path + '/fastsurfer:/output -v ' + output_path + ':/fs_license --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest --fs_license /fs_license/fs_license.txt --t1 /data/WMH_' + t1_image_path.split('/')[-2] + '_0000.nii.gz --sid ' + t1_image_path.split('/')[-2] + ' --sd /output --parallel --seg_only'
    subprocess.run(command, shell=True)
    
    # Create distance maps (to ventricle and corticomedullary junction)
    print('Creating distance maps')
    if not os.path.isdir(output_path + '/distance_maps/'):
        os.mkdir(output_path + '/distance_maps/')
    create_distance_map(output_path + '/fastsurfer/' + t1_image_path.split('/')[-2] + '/mri/aparc.DKTatlas+aseg.deep.mgz', output_path + '/distance_maps/')

    # Spatial subdivision
    print('Performing spatial subdivision')
    if not os.path.isdir(output_path + '/spatial_subdivision/'):
        os.mkdir(output_path + '/spatial_subdivision/')
    if spatial_subdivision == 'connected_components':
        WMH_connected_components(output_path + '/ensemble_preds/WMH_' + t1_image_path.split('/')[-2] +'.nii.gz', output_path + '/distance_maps/' + t1_image_path.split('/')[-2] + '_distance_map_ventricles.nii.gz', output_path + '/spatial_subdivision/')
    elif spatial_subdivision == '10_mm_dilation':
        WMH_dilation_mask_10mm(output_path + '/ensemble_preds/WMH_' + t1_image_path.split('/')[-2] +'.nii.gz', output_path + '/distance_maps/' + t1_image_path.split('/')[-2] + '_distance_map_ventricles.nii.gz', output_path + '/spatial_subdivision/')
    elif spatial_subdivision == 'four_categories':
        WMH_four_categories(output_path + '/ensemble_preds/WMH_' + t1_image_path.split('/')[-2] +'.nii.gz', output_path + '/distance_maps/' + t1_image_path.split('/')[-2] + '_distance_map_ventricles.nii.gz', output_path + '/distance_maps/' + t1_image_path.split('/')[-2] + '_distance_map_cort_junction.nii.gz', output_path + '/spatial_subdivision/')
    
    # T1-intensity based subdivision
    print('Performing T1-signal intensity based subdivision')
    if not os.path.isdir(output_path + '/t1_subdivision/'):
        os.mkdir(output_path + '/t1_subdivision/')
    intensity_subdivision(output_path + '/final_preprocessed_images/WMH_' + t1_image_path.split('/')[-2] + '_0000.nii.gz', output_path + '/t1_subdivision/')

    print('Done!')

if __name__ == '__main__':
    start_time = time.time()
    t1_image_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/data/test_before_preprocessing/160/3DT1.nii.gz'
    flair_image_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/data/test_before_preprocessing/160/FLAIR.nii.gz'
    output_path = '/home/neuro_nph/Desktop/WMH_segmentation_2/data/output_test_2/'
    WMH_segmentation(t1_image_path, flair_image_path, output_path, spatial_subdivision='connected_components')
    end_time = time.time()
    print(end_time - start_time, 'seconds')
