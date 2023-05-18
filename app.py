import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from network import RED_CNN

# Define the preprocessing functions

def load_scan(file):
    try:
        dcm_file = pydicom.filereader.dcmread(file, force=True)
    except pydicom.errors.InvalidDicomError:
        raise ValueError("Invalid DICOM file format. Only .ima files are supported.")

    slice_thickness = dcm_file.SliceThickness

    return dcm_file, slice_thickness


def get_pixels_hu(slices):
    image = slices.pixel_array
    image = image.astype(np.int16)
    image[image == -2000] = 0
    intercept = slices.RescaleIntercept
    slope = slices.RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image


def adjust_brightness(image, brightness_factor):
    image = image * brightness_factor
    image = np.clip(image, 0.0, 1.0)
    return image


def denoise_ct_image(low_dose_image, brightness_factor, model_path):
    # Load the pre-trained model
    model = RED_CNN(out_ch=96)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Filter out unexpected keys from the checkpoint
    state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}

    # Load the filtered state_dict
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Denoise the low dose CT image
    with torch.no_grad():
        low_dose_image_tensor = torch.from_numpy(low_dose_image).unsqueeze(0).unsqueeze(0)
        low_dose_image_tensor = low_dose_image_tensor.to(device)

        # Select a single slice
        low_dose_image_slice = low_dose_image_tensor[:, :, 9, :, :]

        # Convert the input tensor to the same data type as the model's bias
        low_dose_image_slice = low_dose_image_slice.float()

        denoised_image_tensor = model(low_dose_image_slice)
        denoised_image = denoised_image_tensor.squeeze().cpu().numpy()

    # Adjust brightness of the denoised image
    denoised_image = adjust_brightness(denoised_image, brightness_factor)

    return denoised_image

def main():
    st.title("CT Image Denoising")

    # Upload the low dose CT image
    ima_file = st.file_uploader("Upload Low Dose CT Image (IMA)", type="ima")

    if ima_file is not None:
        # Read the IMA file
        slices, slice_thickness = load_scan(ima_file)
        low_dose_image = get_pixels_hu(slices)
        low_dose_image = normalize_(low_dose_image)

        # Set the device
        device = torch.device('cpu')

        # Define model and checkpoint paths
        model_path = 'REDCNN_90epoch.ckpt'

        # Set brightness factor
        brightness_factor = 1.5

        # Denoise the low dose CT image
        denoised_image = denoise_ct_image(low_dose_image, brightness_factor, model_path)

        # Display the results
        st.subheader("Low Dose CT Image")
        st.image(low_dose_image[9], cmap='gray')

        st.subheader("Denoised CT Image")
        st.image(denoised_image, cmap='gray')


if __name__ == "__main__":
    main()
