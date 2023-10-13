import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import warnings
import argparse
import os
import PySimpleGUI as sg
from io import BytesIO
import base64

warnings.filterwarnings("ignore", category=RuntimeWarning)


def grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')


def plot_image(image_path):
    # Load and process the image
    original_image = imageio.imread(image_path)
    grayscale_ = grayscale(original_image)
    invert = 255 - grayscale_
    blur = scipy.ndimage.gaussian_filter(invert, sigma=10)
    processed_image = dodge(blur, grayscale_)

    # Create a plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(processed_image, cmap="gray")
    ax2.set_title("Transformed Image")
    ax2.axis('off')

    # Save the plot to a BytesIO object
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)

    # Encode the plot as base64
    plot_base64 = base64.b64encode(plot_buffer.read()).decode()

    return plot_base64


def main():
    # Define the PySimpleGUI layout for image selection and transformation
    layout = [
        [sg.Text('Select an image file:')],
        [sg.InputText(key='file_path'), sg.FileBrowse()],
        [sg.Button('Transform Image'), sg.Button('Exit')],
        [sg.Image(key='image_plot', size=(400, 400))]
    ]

    # Create the PySimpleGUI window
    window = sg.Window('Image Processor', layout, finalize=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Transform Image':
            image_path = values['file_path']
            if not os.path.isfile(image_path):
                sg.popup_error(f"The file '{image_path}' does not exist.")
                continue

            # Create the plot and convert it to base64
            plot_base64 = plot_image(image_path)

            # Update the displayed plot
            window['image_plot'].update(data=plot_base64)

    window.close()


if __name__ == "__main__":
    main()
