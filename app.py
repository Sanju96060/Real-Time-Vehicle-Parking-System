from flask import Flask, render_template
import sqlite3
import threading
from ultralytics import YOLO
import cv2
import easyocr
from datetime import datetime
import re
import os

app = Flask(__name__)

# Globals
last_message = ""
detection_running = False

# Cooldown tracking: plate_number -> last processed timestamp
last_seen = {}
COOLDOWN_SECONDS = 10  # ignore same plate for this many seconds after any action

# Models
vehicle_model = YOLO("models\\vehical model.pt")
plate_model = YOLO("models\\numberplate.pt")

reader = easyocr.Reader(['en'])

# -------------------------------
# CLEAN PLATE
# -------------------------------
def clean_plate(text):
    text = text.upper().replace(" ", "")
    text = re.sub(r'[^A-Z0-9]', '', text)

    replacements = {
        "O": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }

    return "".join([replacements.get(c, c) for c in text])

# -------------------------------
# VALIDATE
# -------------------------------
def is_valid_indian_plate(plate):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, plate) is not None

# -------------------------------
# FEE CALCULATION
# -------------------------------
def calculate_fee(entry_time):
    entry = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
    exit_time = datetime.now()

    total_seconds = int((exit_time - entry).total_seconds())

    minutes = total_seconds // 60
    seconds = total_seconds % 60

    minutes = max(1, minutes)
    fee = minutes * 2

    duration = f"{minutes} min {seconds} sec"

    return exit_time.strftime("%Y-%m-%d %H:%M:%S"), duration, fee

# -------------------------------
# DETECTION
# -------------------------------
def run_detection():
    global last_message, detection_running

    conn = sqlite3.connect("database/parking.db", check_same_thread=False)
    cursor = conn.cursor()

    cap = cv2.VideoCapture(0)
    os.makedirs("static/vehicles", exist_ok=True)

    detection_running = True

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Vehicle detection
        vehicle_results = vehicle_model(frame)
        detected_vehicle = "Unknown"

        for r in vehicle_results:
            if len(r.boxes.cls) > 0:
                detected_vehicle = r.names[int(r.boxes.cls[0])]

        # Plate detection
        plate_results = plate_model(frame)

        for result in plate_results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                plate_crop = frame[y1:y2, x1:x2]

                results = reader.readtext(plate_crop)

                for (bbox, text, confidence) in results:

                    if confidence < 0.5:
                        continue

                    plate_number = clean_plate(text)

                    if len(plate_number) < 9 or len(plate_number) > 10:
                        continue

                    if not is_valid_indian_plate(plate_number):
                        continue

                    # -------------------------------
                    # COOLDOWN CHECK
                    # -------------------------------
                    now_ts = datetime.now().timestamp()
                    if plate_number in last_seen:
                        if now_ts - last_seen[plate_number] < COOLDOWN_SECONDS:
                            continue

                    # -------------------------------
                    # GET LAST RECORD
                    # -------------------------------
                    cursor.execute("""
                    SELECT id, entry_time, status FROM parking
                    WHERE plate_number=? ORDER BY id DESC LIMIT 1
                    """, (plate_number,))
                    record = cursor.fetchone()

                    print("Record:", record)

                    # -------------------------------
                    # 🚗 EXIT
                    # -------------------------------
                    if record:
                        record_id, entry_time, status = record

                        print("STATUS:", status)

                        if status and status.strip() == "IN":

                            entry_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")

                            # prevent instant exit
                            if (datetime.now() - entry_dt).total_seconds() < 5:
                                continue

                            exit_time, duration, fee = calculate_fee(entry_time)

                            cursor.execute("""
                            UPDATE parking
                            SET exit_time=?, duration=?, fee=?, status='OUT'
                            WHERE id=?
                            """, (exit_time, duration, fee, record_id))

                            conn.commit()

                            print("UPDATED ROW:", record_id)

                            last_message = f"🚗 Vehicle Exited | 💰 Fee: ₹{fee} ({duration})"
                            print("EXIT:", plate_number)

                            last_seen[plate_number] = now_ts

                            # Draw box for exit
                            label = f"{plate_number} | EXIT"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                            continue   # stop further processing

                    # -------------------------------
                    # 🚗 ENTRY
                    # -------------------------------
                    entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    filename = f"{plate_number}_{int(datetime.now().timestamp())}.jpg"
                    image_path = f"vehicles/{filename}"

                    cv2.imwrite(f"static/{image_path}", frame)

                    cursor.execute("""
                    INSERT INTO parking
                    (plate_number, vehicle_model, entry_time, status, image_path)
                    VALUES (?, ?, ?, 'IN', ?)
                    """, (plate_number, detected_vehicle, entry_time, image_path))

                    conn.commit()

                    last_message = "🚗 Vehicle Entered"
                    print("ENTRY:", plate_number)

                    last_seen[plate_number] = now_ts

                    # Draw box
                    label = f"{plate_number} | {detected_vehicle}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Smart Parking System", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

# -------------------------------
# ROUTE
# -------------------------------
@app.route("/")
def home():
    global last_message

    conn = sqlite3.connect("database/parking.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM parking ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()

    msg = last_message
    last_message = ""

    return render_template("index.html", data=data, message=msg)

# -------------------------------
# START THREAD
# -------------------------------
def start_detection():
    thread = threading.Thread(target=run_detection, daemon=True)
    thread.start()
    print("Detection Started")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    start_detection()
    app.run(debug=True, use_reloader=False)