from ultralytics import YOLO
import cv2
import easyocr
import sqlite3
from datetime import datetime
import re


# -------------------------------
# Load YOLO Models
# -------------------------------

vehicle_model = YOLO("models\\vehical model.pt")
plate_model = YOLO("models\\numberplate.pt")


# -------------------------------
# EasyOCR Setup
# -------------------------------

reader = easyocr.Reader(['en'])


# -------------------------------
# Database Connection
# -------------------------------

conn = sqlite3.connect("database/parking.db")
cursor = conn.cursor()


# -------------------------------
# Indian Number Plate Validation
# -------------------------------



def is_valid_indian_plate(plate):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, plate) is not None

# -------------------------------
# Camera Start
# -------------------------------

cap = cv2.VideoCapture(0)

last_detected_plate = ""


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # -------------------------------
    # Detect Vehicle Model
    # -------------------------------

    vehicle_results = vehicle_model(frame)

    detected_vehicle = "Unknown"

    for r in vehicle_results:

        if len(r.boxes.cls) > 0:

            detected_vehicle = r.names[int(r.boxes.cls[0])]


    # -------------------------------
    # Detect Number Plate
    # -------------------------------

    plate_results = plate_model(frame)


    for result in plate_results:

        for box in result.boxes.xyxy:

            x1, y1, x2, y2 = map(int, box)

            plate_crop = frame[y1:y2, x1:x2]


            # -------------------------------
            # OCR Read Plate Text
            # -------------------------------

            text = reader.readtext(plate_crop)


            if text:

                plate_number = text[0][-2]

                # Remove spaces and convert to uppercase
                plate_number = plate_number.replace(" ", "").upper()


                # -------------------------------
                # Validate Indian Plate Format
                # -------------------------------

                if is_valid_indian_plate(plate_number):

                    # Prevent duplicate detection
                    if plate_number != last_detected_plate:

                        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


                        cursor.execute("""
                        INSERT INTO parking
                        (plate_number, vehicle_model, entry_time)
                        VALUES (?, ?, ?)
                        """, (plate_number, detected_vehicle, entry_time))


                        conn.commit()


                        last_detected_plate = plate_number


                        print("Saved Successfully:")
                        print("Plate:", plate_number)
                        print("Vehicle:", detected_vehicle)
                        print("Entry Time:", entry_time)
                        print("---------------------")


                else:

                    print("Invalid Plate Ignored:", plate_number)


                # -------------------------------
                # Draw Detection Box on Screen
                # -------------------------------

                label = f"{plate_number} | {detected_vehicle}"


                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0),
                              2)


                cv2.putText(frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)


    # -------------------------------
    # Show Camera Output
    # -------------------------------

    cv2.imshow("Smart Parking Entry Detection", frame)


    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()

conn.close()