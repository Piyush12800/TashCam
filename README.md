# TashCam

**TashCam** is a virtual camera application that creates dynamic color streams using the HSV color model. This project allows users to utilize a virtual camera in video conferencing applications, providing an engaging and colorful background.

## Features

- Generates a continuously changing color frame based on the HSV color model.
- Easy to set up and use with any video conferencing application (Zoom, Teams, etc.).
- Customizable resolution and frame rate.

## Requirements

- Python 3.7 or higher
- `opencv-python`
- `numpy`
- `pyvirtualcam`
- `colorsys`
- **OBS Virtual Camera** or **UnityCapture** for virtual camera output

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tashcam.git
   cd tashcam ```

2 Requirements 
    ```bash
    pip install opencv-python numpy pyvirtualcam ```
3. Set up OBS Studio or UnityCapture:

OBS Studio: Download and install from obsproject.com.
UnityCapture: Ensure it's installed and configured to use your virtual camera.

## Usage
Run the application:

bash
```
python app.py
