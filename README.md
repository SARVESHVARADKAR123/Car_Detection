# Car and People Detection System

This project is a real-time detection system that identifies cars and people in images or video streams. It uses YOLOv8 for object detection and a custom Keras model for car color classification.

## Features

- **Real-time Detection**: Detects multiple cars and people in a frame.
- **Color Classification**: Classifies car colors and highlights blue cars with red boxes.
- **Statistics**: Displays the count of cars, people, and car colors.
- **GUI Interface**: Built with Tkinter for easy interaction.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Models**:
   - [YOLOv8 Weights](https://drive.google.com/file/d/16KY0KchxGYSQjcQMfBEGZfTFM8VX2bon/view?usp=drive_link)
   - [Custom Keras Model](https://drive.google.com/file/d/1YvIj7kHzi0OdZUoMXo8T62qcf_w3C447/view?usp=drive_link)
   - [Pickle File of model](https://drive.google.com/file/d/1sc8MpJ3h4fY3uTtZo2mtFMpNCu_3ceiT/view?usp=drive_link)
   - [Weights](https://drive.google.com/file/d/1eAPCJAf_tiLUknIz2lppqmSu1e6nbmO7/view?usp=drive_link)

## Usage

1. **Run the Application**:
   ```bash
   python car_detection_app.py
   ```

2. **Features**:
   - **Open Image**: Select an image file to process.
   - **Open Camera**: Use your webcam for real-time detection.
   - **Reset Stats**: Clear the current statistics.

## File Structure

- `car_detection_app.py`: Main application file.
- `requirements.txt`: List of dependencies.
- `.gitignore`: Specifies files to be ignored by Git.

## Configuration

- **YOLOv8 Model**: Ensure `yolov8n.pt` is in the directory.
- **Color Model**: Place your custom Keras model `car_color_custom_model.keras` in the directory.

## Google Colab

- [Colab Notebook](https://colab.research.google.com/drive/1Sk73VhAQlBqoILHX6X1rgUo6wgshwUc4?usp=drive_link)

