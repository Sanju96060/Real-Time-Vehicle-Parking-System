An automated parking management system that uses YOLOv8 for vehicle and license plate detection, EasyOCR for reading Indian number plates, and a Flask + SQLite backend to log vehicle entries/exits and calculate parking fees in real time from a live camera feed.

Features


🔍 Vehicle Detection — Classifies vehicle type (car, bus, motorcycle, pickup-truck, semi-trailer, van) using a custom-trained YOLOv8 model.
🔢 License Plate Detection & OCR — Detects number plates with YOLOv8 and reads them using EasyOCR.
🇮🇳 Indian Plate Validation — Cleans OCR output (common misreads like O→0, I→1) and validates against the standard Indian plate format (XX00XX0000).
🚪 Automatic Entry/Exit Logging — Detects whether a plate is entering or exiting based on existing database records.
💰 Automatic Fee Calculation — Computes parking duration and fee on exit.
🕒 Cooldown Logic — Prevents duplicate entries from repeated detections of the same plate within a short window.
🖥️ Web Dashboard — Flask route displaying all parking records with live status messages.
📸 Vehicle Snapshots — Saves an image of each vehicle at entry.

.
├── app.py                     # Main Flask app + real-time detection loop
├── entry_detect.py            # Standalone script for entry-only detection (early/dev version)
├── database.py                # Database schema setup (parking table)
├── YOLO_Training_Colab.ipynb  # Colab notebook to train the vehicle & plate YOLO models
├── yolov8n.pt                  # Base YOLOv8 nano weights used as a starting point for training
├── models/
│   ├── vehical model.pt       # Trained vehicle detection model (custom)
│   └── numberplate.pt         # Trained number plate detection model (custom)
├── database/
│   └── parking.db             # SQLite database (created at runtime)
├── static/
│   └── vehicles/              # Saved snapshots of vehicles at entry
└── templates/
    └── index.html              # Dashboard template (not included — add your own)

How It Works

A live camera feed is captured with OpenCV.
Each frame is run through the vehicle detection model to identify the vehicle type.
Each frame is also run through the plate detection model to locate number plates.
Detected plate regions are cropped and passed to EasyOCR for text extraction.
The extracted text is cleaned and validated as a proper Indian plate number.
The system checks the database for an existing "IN" record for that plate:

No existing record → logs a new entry with timestamp, vehicle type, and snapshot.
Existing "IN" record → logs an exit, calculates duration and fee, and updates the record.

A cooldown timer prevents the same plate from being processed multiple times in quick succession.
The Flask dashboard (/) displays all records and the latest entry/exit message.
