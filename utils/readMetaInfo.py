import nibabel as nib
import pydicom
import matplotlib.pyplot as plt

BASE_IMG = 'data/PublicAbdominalData/01_Multi-Atlas_Labeling/img/'
file = 'img0067.nii.gz'

def readHeader(file = file, BASE_IMG = BASE_IMG):
# Load the NIfTI file
    for i in range(61, 80, 1):
        try:
            file = f'img00{str(i)}.nii.gz'
            nii_file = BASE_IMG + file
            img = nib.load(nii_file)

            # Get the header
            header = img.header

            # Print out the metadata
            print(f"idx{i}: glmax: {header['glmax']}, glmin: {header['glmin']}")
        except:
            print("cannot open or find the file")

    # decrypted the metainfo
    '''
    bad segmentation for img0069.nii.gz
    idx61: glmax: 1684, glmin: -1024
    idx62: glmax: 1618, glmin: -2048
    idx63: glmax: 1500, glmin: -3024
    idx64: glmax: 2976, glmin: -1024
    idx65: glmax: 2976, glmin: -1024
    idx66: glmax: 1427, glmin: -1024
    idx67: glmax: 1278, glmin: -1024
    idx68: glmax: 1550, glmin: -3024
    idx69: glmax: 9580, glmin: -2048
    idx70: glmax: 2976, glmin: -1024
    idx74: glmax: 3071, glmin: -3024
    idx75: glmax: 3071, glmin: -3024
    idx76: glmax: 1440, glmin: -1024
    idx77: glmax: 2976, glmin: -1024
    idx78: glmax: 3071, glmin: -3024
    idx79: glmax: 3071, glmin: -1024
    '''

def readIntensity(file = file, BASE_IMG = BASE_IMG):

    for i in range(1, 41, 1):
        try:
            # file = f'img00{str(i)}.nii.gz'
            # nii_file = BASE_IMG + file
            file_name = f'img000{str(i)}.nii.gz' if i < 10 else f'img00{str(i)}.nii.gz'
            nii_file = f'D:/AbdomenAtlas_CP/data/PublicAbdominalData/01_Multi-Atlas_Labeling/img/'+file_name
            nifti_image = nib.load(nii_file)

            # Get the image data from the NIfTI file object
            image_data = nifti_image.get_fdata()
            # Flatten the 3D array to 1D for histogram computation
            flattened_data = image_data.flatten()

            x_min = -2100
            x_max = 1500
            
            # Plotting the histogram
            if i == 69:
                plt.hist(flattened_data, bins=50, alpha=1, range=[flattened_data.min(), flattened_data.max()], label=file_name)
            else: plt.hist(flattened_data, bins=50, alpha=0.3, range=[flattened_data.min(), flattened_data.max()], label=file_name)
            
        except:
            print("cannot open or find the file")
            
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.show()
    

if __name__ == "__main__":
    readIntensity()