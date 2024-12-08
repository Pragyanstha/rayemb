import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", type=str, default="data/CTPelvic1K/dataset6_volume/dataset6_CLINIC_0001_data.nii.gz")
# parser.add_argument("--mask", type=str, default="data/CTPelvic1K/dataset6_label/dataset6_CLINIC_0001_mask_4label.nii.gz")
parser.add_argument("--factor", type=int, default=2)
parser.add_argument("--threshold", type=float, default=250)
parser.add_argument("--output", type=str, default="ui/public/test.nii.gz")
args = parser.parse_args()

def main(args):
    img = nib.load(args.input)
    # mask = nib.load(args.mask)
    # Get the data and affine
    data = img.get_fdata()
    # mask_data = mask.get_fdata()
    # data[mask_data == 0] = 0
    data[data <= args.threshold] = 0
    affine = img.affine
    header = img.header

    # Downsize the volume data (factor of 0.5 in each dimension)
    zoom_factors = [1/args.factor, 1/args.factor, 1/args.factor]
    downsampled_data = zoom(data, zoom_factors, order=3)  # Cubic interpolation
    # also mask out the outside region
    # downsampled_data[downsampled_data <= args.threshold] = 0
    # Update the affine matrix
    new_affine = affine.copy()
    new_affine[:3, :3] *= args.factor  # Scale voxel size by args.factor (inverse of zoom factor)

    # Update header dimensions
    new_header = header.copy()
    new_header["dim"][1:4] = downsampled_data.shape  # Update dimensions
    new_header["pixdim"][1:4] *= args.factor  # Adjust voxel size accordingly

    # Create the downsized NIfTI image
    downsampled_img = nib.Nifti1Image(downsampled_data, new_affine, header=new_header)
    nib.save(downsampled_img, args.output)

if __name__ == "__main__":
    main(args)