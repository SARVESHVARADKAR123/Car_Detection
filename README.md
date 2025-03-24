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
   - Ensure `yolov8n.pt` and `my_model.keras` are in the project directory.

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
- **Color Model**: Place your custom Keras model `my_model.keras` in the directory.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.