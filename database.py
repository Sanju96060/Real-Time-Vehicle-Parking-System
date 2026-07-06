# import sqlite3

# conn = sqlite3.connect("database/parking.db")

# conn.execute("""
# CREATE TABLE IF NOT EXISTS parking(
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     plate_number TEXT UNIQUE,
#     vehicle_model TEXT,
#     entry_time TEXT,
#     image_path TEXT   -- 🔥 NEW COLUMN
# )
# """)

# conn.close()