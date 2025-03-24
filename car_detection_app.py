import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

class CarDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car and People Detection System")
        
        # Load models
        self.yolo_model = YOLO('yolov8n.pt')
        self.color_model = tf.keras.models.load_model('car_color_custom_model.keras')
        
        # Define colors list in correct order
        self.colors = [
            'beige', 'black', 'blue', 'brown', 'gold',
            'green', 'grey', 'orange', 'pink', 'purple',
            'red', 'silver', 'tan', 'white', 'yellow'
        ]
        
        # Initialize statistics
        self.color_stats = {color: 0 for color in self.colors}
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(button_frame, text="Open Image", command=self.open_image).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Open Camera", command=self.open_camera).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Reset Stats", command=self.reset_stats).grid(row=0, column=2, padx=5)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(main_frame, width=800, height=600, bg='black')
        self.canvas.grid(row=1, column=0, pady=10)
        
        # Create statistics panel
        stats_frame = ttk.LabelFrame(main_frame, text="Detection Statistics", padding="5")
        stats_frame.grid(row=1, column=1, padx=10, sticky="n")
        
        # Car and people counters
        self.car_count_var = tk.StringVar(value="Cars: 0")
        self.people_count_var = tk.StringVar(value="People: 0")
        ttk.Label(stats_frame, textvariable=self.car_count_var, font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5)
        ttk.Label(stats_frame, textvariable=self.people_count_var, font=('Arial', 12, 'bold')).grid(row=1, column=0, pady=5)
        
        # Color statistics
        ttk.Label(stats_frame, text="Car Colors:", font=('Arial', 10, 'bold')).grid(row=2, column=0, pady=(10,5), sticky="w")
        self.color_vars = {}
        for i, color in enumerate(self.colors):
            self.color_vars[color] = tk.StringVar(value=f"{color.title()}: 0")
            ttk.Label(stats_frame, textvariable=self.color_vars[color]).grid(row=i+3, column=0, padx=5, sticky="w")
        
        self.camera = None
        self.is_camera_active = False
    
    def reset_stats(self):
        """Reset all statistics"""
        self.color_stats = {color: 0 for color in self.colors}
        self.update_stats(0, 0)
    
    def update_stats(self, car_count, people_count):
        """Update statistics display"""
        self.car_count_var.set(f"Cars: {car_count}")
        self.people_count_var.set(f"People: {people_count}")
        for color in self.colors:
            self.color_vars[color].set(f"{color.title()}: {self.color_stats[color]}")
    
    def preprocess_for_color(self, car_crop):
        """Preprocess image for color classification"""
        IMG_SIZE = (224, 224)
        # Convert to RGB if needed
        if len(car_crop.shape) == 2:  # If grayscale
            car_crop = cv2.cvtColor(car_crop, cv2.COLOR_GRAY2RGB)
        elif car_crop.shape[2] == 4:  # If RGBA
            car_crop = cv2.cvtColor(car_crop, cv2.COLOR_RGBA2RGB)
        
        # Resize to match model's input size
        car_crop_resized = cv2.resize(car_crop, IMG_SIZE)
        # Normalize pixel values
        car_crop_normalized = car_crop_resized / 255.0
        # Add batch dimension
        car_crop_batch = np.expand_dims(car_crop_normalized, axis=0)
        return car_crop_batch
    
    def process_image(self, frame):
        """Process a single frame/image"""
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = self.yolo_model(frame_rgb)
        
        # Statistics counters
        car_count = 0
        people_count = 0
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class and confidence
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Class 2 is car in COCO dataset
                if cls == 2 and conf > 0.3:  # Car with confidence threshold
                    car_count += 1
                    
                    # Crop car region
                    car_crop = frame_rgb[y1:y2, x1:x2]
                    
                    try:
                        # Preprocess for color classification
                        preprocessed = self.preprocess_for_color(car_crop)
                        
                        # Get color prediction
                        color_pred = self.color_model.predict(preprocessed, verbose=0)
                        color_idx = np.argmax(color_pred[0])
                        color_name = self.colors[color_idx]
                        
                        # Update color statistics
                        self.color_stats[color_name] += 1
                        
                        # Draw bounding box - Red for blue cars, Blue for others
                        if color_name == 'blue':
                            box_color = (255, 0, 0)  # Red in RGB
                            text_color = (255, 0, 0)  # Red text
                        else:
                            box_color = (0, 0, 255)  # Blue in RGB
                            text_color = (0, 0, 255)  # Blue text
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), box_color, 2)
                        label = f"{color_name} car {conf:.2f}"
                        # Draw label background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame_rgb, (x1, y1-25), (x1 + label_size[0], y1), box_color, -1)
                        # Draw label text
                        cv2.putText(frame_rgb, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                    except Exception as e:
                        print(f"Error processing car crop: {e}")
                        continue
                
                # Class 0 is person in COCO dataset
                elif cls == 0 and conf > 0.3:  # Person with confidence threshold
                    people_count += 1
                    # Draw green box for people
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add person label
                    cv2.putText(frame_rgb, f"Person {conf:.2f}", (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Update statistics
        self.update_stats(car_count, people_count)
        
        # Convert to PhotoImage for display
        height, width = frame_rgb.shape[:2]
        scale = min(800/width, 600/height)
        new_width, new_height = int(width * scale), int(height * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        
        # Update canvas
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
    
    def open_image(self):
        """Open and process an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.process_image(image)
    
    def open_camera(self):
        """Toggle camera on/off"""
        if not self.is_camera_active:
            self.camera = cv2.VideoCapture(0)
            self.is_camera_active = True
            self.update_camera()
        else:
            if self.camera is not None:
                self.camera.release()
            self.is_camera_active = False
    
    def update_camera(self):
        """Update camera feed"""
        if self.is_camera_active and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                self.process_image(frame)
                self.root.after(10, self.update_camera)

if __name__ == "__main__":
    root = tk.Tk()
    app = CarDetectionApp(root)
    root.mainloop() 