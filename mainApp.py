# mainApp.py
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QSizePolicy
from pyzbar.pyzbar import decode as qr_decode
import qrcode
import numpy as np

APP_NAME = "Airport Luggage Management — AR"
DB_FILE = "airportluggage.db"
QR_DIR = Path("qrs")
QR_DIR.mkdir(exist_ok=True)

# --------------------- Belt routing ---------------------
DEST_MAP = {
    "LONDON": {"area": "B", "belt": "12"},
    "DOHA": {"area": "A", "belt": "3"},
    "DUBAI": {"area": "D", "belt": "7"},
    "PARIS": {"area": "C", "belt": "5"},
    "SINGAPORE": {"area": "E", "belt": "9"},
    "NEWYORK": {"area": "F", "belt": "14"},
}
def compute_routing(destination: str) -> tuple[str, str]:
    key = (destination or "").strip().upper()
    if key in DEST_MAP:
        m = DEST_MAP[key]
        return m["area"], m["belt"]
    seed = sum(ord(c) for c in key)
    areas = "ABCDEFG"
    return areas[seed % len(areas)], str((seed % 15) + 1)

# --------------------- Data / DB ---------------------
@dataclass
class Luggage:
    id: str
    first: str
    last: str
    destination: str
    flight_no: str
    belt_area: str
    belt_no: str
    created_at: str
    qr_path: str

class DB:
    def __init__(self, path=DB_FILE):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS luggage(
                id TEXT PRIMARY KEY,
                first TEXT NOT NULL,
                last TEXT NOT NULL,
                destination TEXT NOT NULL,
                flight_no TEXT NOT NULL,
                belt_area TEXT NOT NULL,
                belt_no TEXT NOT NULL,
                created_at TEXT NOT NULL,
                qr_path TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, lug: Luggage):
        self.conn.execute(
            "INSERT INTO luggage VALUES (?,?,?,?,?,?,?,?,?)",
            (lug.id, lug.first, lug.last, lug.destination, lug.flight_no,
             lug.belt_area, lug.belt_no, lug.created_at, lug.qr_path)
        )
        self.conn.commit()

    def get(self, id_: str) -> Luggage | None:
        cur = self.conn.execute(
            "SELECT id,first,last,destination,flight_no,belt_area,belt_no,created_at,qr_path "
            "FROM luggage WHERE id=?",
            (id_.strip(),),
        )
        row = cur.fetchone()
        return Luggage(*row) if row else None

    def delete(self, id_: str) -> bool:
        cur = self.conn.execute("DELETE FROM luggage WHERE id=?", (id_.strip(),))
        self.conn.commit()
        return cur.rowcount > 0

# --------------------- QR helpers ---------------------
def make_qr_png(payload: dict, out_path: Path) -> Path:
    qrcode.make(json.dumps(payload, separators=(",", ":"))).save(out_path)
    return out_path

# --------------------- Camera / Scanner with AR + Recording ---------------------
class CameraScanner(QtWidgets.QDialog):
    """
    - Detects QR -> emits parsed payload via `scanned` (updates main UI).
    - Detects ArUco markers -> true AR: draws 3D axis + cube anchored to marker.
    - Records the camera session to MP4 while the dialog is open.
    """
    scanned = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan from Camera — AR")
        self.resize(920, 580)
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background:#000;color:#fff; border-radius:10px;")
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.close_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        # --- ArUco setup (opencv-contrib-python required) ---
        self.aruco = cv2.aruco
        self.aruco_dict = self.aruco.getPredefinedDictionary(self.aruco.DICT_4X4_50)
        self.aruco_params = self.aruco.DetectorParameters_create()

        # Marker size in meters (adjust to your printed marker)
        self.marker_length_m = 0.04  # 4 cm

        # Camera intrinsics (set on first frame if not calibrated)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Recording
        self.writer = None
        self.fps = 30
        self.record_path = "ar_session.mp4"

    def start(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(
                self, "Camera Error",
                "<span style='color:#000000;'>Cannot open camera.</span>"
            )
            self.reject()
            return

        # Setup video writer
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.record_path, fourcc, self.fps, (w, h))

        self.timer.start(1000 // self.fps)
        self.exec()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        event.accept()

    # --- simple intrinsics estimate (fallback) ---
    def _estimate_camera_matrix(self, w, h):
        # Rough intrinsics if you don't have calibration: ~0.9*w focal guess
        f = 0.9 * w
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((5, 1), dtype=np.float32)
        return K, dist

    def _draw_axis(self, frame, K, dist, rvec, tvec, length=0.03):
        axis = np.float32([[0, 0, 0],
                           [length, 0, 0],
                           [0, length, 0],
                           [0, 0, length]])
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
        p0 = tuple(imgpts[0].ravel().astype(int))
        cv2.line(frame, p0, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3)    # X red
        cv2.line(frame, p0, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 3)    # Y green
        cv2.line(frame, p0, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 3)    # Z blue

    def _draw_cube(self, frame, K, dist, rvec, tvec, size=0.03):
        s = size
        pts = np.float32([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]
        ])
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        # base
        for i, j in zip([0,1,2,3], [1,2,3,0]):
            cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 200, 0), 2)
        # pillars
        for i in range(4):
            cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i+4]), (255, 200, 0), 2)
        # top
        for i, j in zip([4,5,6,7], [5,6,7,4]):
            cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 200, 0), 2)

    def _next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        h, w = frame.shape[:2]
        if self.camera_matrix is None:
            self.camera_matrix, self.dist_coeffs = self._estimate_camera_matrix(w, h)

        # ---- QR decode (updates main UI) ----
        decoded = qr_decode(frame)
        for c in decoded:
            try:
                payload = json.loads(c.data.decode("utf-8"))
                (x, y, wq, hq) = c.rect
                cv2.rectangle(frame, (x, y), (x+wq, y+hq), (60, 220, 255), 2)
                cv2.putText(frame, "QR OK", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,220,255), 2)
                # emit and close (same behavior as before)
                self.scanned.emit(payload)
                # Write the "success" frame once more for the recording, then close
                if self.writer is not None:
                    self.writer.write(frame)
                self.close()
                return
            except Exception:
                pass

        # ---- ArUco detection + pose (true AR overlay) ----
        corners, ids, _ = self.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None and len(ids):
            self.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = self.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length_m, self.camera_matrix, self.dist_coeffs
            )
            for rvec, tvec in zip(rvecs, tvecs):
                self._draw_axis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, length=0.03)
                self._draw_cube(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, size=0.03)
            cv2.putText(frame, "ArUco detected: 3D overlay active",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.putText(frame, "Point at a QR (fills details) or ArUco marker (3D overlay).",
                        (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,230), 2)

        # Write to video file
        if self.writer is not None:
            self.writer.write(frame)

        # Show in dialog
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

# --------------------- Main Window (design & behavior preserved) ---------------------
class Main(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1150, 720)

        self.db = DB()
        self.current_record: Luggage | None = None

        # THEME
        self.setObjectName("root")
        self.setStyleSheet("""
        QWidget { color:#0b1324; font-size:15px; }

        #root {
          background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                      stop:0 #800020, stop:0.55 #143a63, stop:1 #0ea5e9);
        }

        QFrame.card {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ffffff, stop:1 #f6faff);
            border: 2px solid #c8d5f0;
            border-radius: 14px;
        }

        QLabel.cardTitle {
            color:#0b2a55; font-weight:900; letter-spacing:.3px;
            padding:2px 8px; border-radius:6px; background:transparent;
        }

        QLineEdit {
            background:#ffffff; color:#0b1324;
            border:1px solid #c8d5f0; border-radius:10px; padding:10px 12px;
        }
        QLineEdit::placeholder { color:#6b7a93; }
        QLineEdit[readonly="true"] { background:#f3f6fb; color:#334155; }

        QPushButton {
            background:#0ea5e9; border:1px solid #0ea5e9;
            color:white; border-radius:10px; padding:10px 16px; font-weight:800;
        }
        QPushButton:hover { background:#0284c7; }
        QPushButton[secondary="true"] { background:#e2e8f0; color:#111827; border-color:#d0d7e2; }
        QPushButton[secondary="true"]:hover { background:#d9e0ea; }
        QPushButton[danger="true"] { background:#ef4444; border-color:#ef4444; color:white; }

        QLabel.fieldLabel { color:#0b2a55; font-weight:900; }
        QLabel#qr { background:#fff; border:2px dashed #cbd5e1; border-radius:14px; padding:14px; }

        QLabel[key="true"] { color:#0b2a55; font-weight:900; }
        QLabel[val="true"] { color:#0b1324; font-weight:900; }
        """)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#6b7a93"))
        self.setPalette(pal)

        # OUTER
        root = QtWidgets.QFrame(); root.setProperty("class", "card")
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(root)
        rootBox = QtWidgets.QVBoxLayout(root); rootBox.setContentsMargins(20, 20, 20, 20); rootBox.setSpacing(18)

        # Header
        title = QtWidgets.QLabel("AIRPORT LUGGAGE DASHBOARD")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size:28px; font-weight:900; letter-spacing:1px; color:#0b2a55;")
        rootBox.addWidget(title)

        # Grid area
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(16)
        rootBox.addLayout(grid)

        # ---------- LEFT COLUMN ----------
        leftWrap = QtWidgets.QVBoxLayout(); leftWrap.setSpacing(14)

        leftCard = QtWidgets.QFrame(); leftCard.setProperty("class", "card")
        leftBox = QtWidgets.QVBoxLayout(leftCard); leftBox.setContentsMargins(16, 16, 16, 16); leftBox.setSpacing(10)

        leftHeader = self._section_header("Add Luggage")
        leftBox.addLayout(leftHeader)

        form = QtWidgets.QGridLayout(); form.setHorizontalSpacing(12); form.setVerticalSpacing(12)
        def lab(text):
            lb = QtWidgets.QLabel(text); lb.setProperty("class", "fieldLabel")
            lb.setMinimumWidth(140)
            lb.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            return lb

        self.flight = QtWidgets.QLineEdit(placeholderText="Enter flight number")
        self.first  = QtWidgets.QLineEdit(placeholderText="Enter first name")
        self.last   = QtWidgets.QLineEdit(placeholderText="Enter last name")
        self.dest   = QtWidgets.QLineEdit(placeholderText="Enter destination")
        self.add_btn = QtWidgets.QPushButton("Add Luggage")
        self.new_btn = QtWidgets.QPushButton("New Entry"); self.new_btn.setProperty("secondary", True)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.new_btn)
        btn_row.addStretch(1)

        form.addWidget(lab("Flight No."), 0, 0); form.addWidget(self.flight, 0, 1)
        form.addWidget(lab("First Name"), 1, 0); form.addWidget(self.first, 1, 1)
        form.addWidget(lab("Last Name"),  2, 0); form.addWidget(self.last,  2, 1)
        form.addWidget(lab("Destination"),3, 0); form.addWidget(self.dest,  3, 1)
        form.addLayout(btn_row,            4, 1)
        leftBox.addLayout(form)

        # Bottom Luggage ID search card
        idCard = QtWidgets.QFrame(); idCard.setProperty("class", "card")
        idBox = QtWidgets.QVBoxLayout(idCard); idBox.setContentsMargins(16, 16, 16, 16); idBox.setSpacing(10)
        idHeader = self._section_header("Luggage ID")
        idBox.addLayout(idHeader)

        idGrid = QtWidgets.QGridLayout(); idGrid.setHorizontalSpacing(10); idGrid.setVerticalSpacing(10)
        self.search_id = QtWidgets.QLineEdit(placeholderText="Enter luggage ID")
        self.search_btn = QtWidgets.QPushButton("Search"); self.search_btn.setProperty("secondary", True)
        self.delete_btn = QtWidgets.QPushButton("Delete"); self.delete_btn.setProperty("danger", True)
        idGrid.addWidget(self.search_id, 0, 0, 1, 2)
        idGrid.addWidget(self.search_btn, 1, 0)
        idGrid.addWidget(self.delete_btn, 1, 1)
        idBox.addLayout(idGrid)

        leftWrap.addWidget(leftCard)
        leftWrap.addWidget(idCard)
        leftContainer = QtWidgets.QWidget(); leftContainer.setLayout(leftWrap)
        grid.addWidget(leftContainer, 0, 0)

        # ---------- RIGHT COLUMN ----------
        rightCard = QtWidgets.QFrame(); rightCard.setProperty("class", "card")
        rightBox = QtWidgets.QVBoxLayout(rightCard); rightBox.setContentsMargins(16, 16, 16, 16); rightBox.setSpacing(10)
        rightHeader = self._section_header("QR Details")
        rightBox.addLayout(rightHeader)

        self.qr_label = QtWidgets.QLabel(objectName="qr")
        self.qr_label.setMinimumSize(300, 300)
        self.qr_label.setMaximumSize(300, 300)
        self.qr_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.qr_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rightBox.addWidget(self.qr_label, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        btnRow = QtWidgets.QHBoxLayout(); btnRow.setSpacing(12)
        self.save_qr_btn = QtWidgets.QPushButton("Save QR"); self.save_qr_btn.setProperty("secondary", True)
        self.scan_btn = QtWidgets.QPushButton("Scan from Camera")
        btnRow.addWidget(self.save_qr_btn); btnRow.addWidget(self.scan_btn); btnRow.addStretch(1)
        rightBox.addLayout(btnRow)

        details = QtWidgets.QGridLayout(); details.setHorizontalSpacing(12); details.setVerticalSpacing(10)
        details.setColumnStretch(0, 0)
        details.setColumnStretch(1, 1)
        details.setColumnStretch(2, 0)

        def key(text):
            k = QtWidgets.QLabel(text); k.setProperty("key", True); return k
        def val(text="N/A"):
            v = QtWidgets.QLabel(text); v.setProperty("val", True)
            v.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            return v

        self.v_flight = val()
        self.v_area   = val()
        self.v_belt   = val()
        self.v_id     = val()

        details.addWidget(key("Flight No:"),   0, 0); details.addWidget(QtWidgets.QWidget(), 0, 1); details.addWidget(self.v_flight, 0, 2)
        details.addWidget(key("Belt Area:"),   1, 0); details.addWidget(QtWidgets.QWidget(), 1, 1); details.addWidget(self.v_area,   1, 2)
        details.addWidget(key("Belt Number:"), 2, 0); details.addWidget(QtWidgets.QWidget(), 2, 1); details.addWidget(self.v_belt,   2, 2)
        details.addWidget(key("Luggage ID:"),  3, 0); details.addWidget(QtWidgets.QWidget(), 3, 1); details.addWidget(self.v_id,     3, 2)

        rightBox.addLayout(details)
        grid.addWidget(rightCard, 0, 1)

        # connections
        self.add_btn.clicked.connect(self.add_luggage)
        self.new_btn.clicked.connect(self.reset_cards)
        self.search_btn.clicked.connect(self.search_luggage)
        self.delete_btn.clicked.connect(self.delete_luggage)
        self.save_qr_btn.clicked.connect(self.save_current_qr)
        self.scan_btn.clicked.connect(self.scan_camera)

    def _section_header(self, text: str) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout(); row.setSpacing(8)
        lbl = QtWidgets.QLabel(text); lbl.setProperty("class", "cardTitle")
        rule = QtWidgets.QFrame(); rule.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        rule.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        rule.setStyleSheet("color:#c8d5f0;")
        row.addWidget(lbl); row.addWidget(rule, 1)
        return row

    # visible alert helpers
    def show_info(self, title, message_html):
        QtWidgets.QMessageBox.information(self, title, f"<span style='color:#000000;'>{message_html}</span>")
    def show_warning(self, title, message_html):
        QtWidgets.QMessageBox.warning(self, title, f"<span style='color:#000000;'>{message_html}</span>")
    def show_critical(self, title, message_html):
        QtWidgets.QMessageBox.critical(self, title, f"<span style='color:#000000;'>{message_html}</span>")

    def reset_cards(self, keep_search: bool = True):
        self.flight.clear(); self.first.clear(); self.last.clear(); self.dest.clear()
        if not keep_search: self.search_id.clear()
        self.v_flight.setText("N/A"); self.v_area.setText("N/A"); self.v_belt.setText("N/A"); self.v_id.setText("N/A")
        self.qr_label.clear()
        self.flight.setFocus()

    # -------- actions --------
    def add_luggage(self):
        first = self.first.text().strip()
        last = self.last.text().strip()
        dest = self.dest.text().strip().upper()
        flight = self.flight.text().strip().upper()

        if not all([first, last, dest, flight]):
            self.show_warning("Missing", "Please fill First/Title, Last, Destination, and Flight No.")
            return

        belt_area, belt_no = compute_routing(dest)
        lug_id = uuid.uuid4().hex[:8]
        payload = {
            "id": lug_id, "first": first, "last": last, "destination": dest,
            "flightNo": flight, "beltArea": belt_area, "beltNo": belt_no,
            "ts": datetime.now(UTC).isoformat(),
        }

        qr_path = QR_DIR / f"luggage_{lug_id}.png"
        make_qr_png(payload, qr_path)

        lug = Luggage(
            id=lug_id, first=first, last=last, destination=dest, flight_no=flight,
            belt_area=belt_area, belt_no=belt_no, created_at=datetime.now(UTC).isoformat(),
            qr_path=str(qr_path)
        )
        self.db.add(lug)
        self._update_display(lug)

        self.show_info("Success", f"Luggage Added Successfully!<br>ID: {lug_id}")
        # No auto-clear; use New Entry when you're ready.

    def search_luggage(self):
        id_ = self.search_id.text().strip()
        if not id_:
            return
        lug = self.db.get(id_)
        if not lug:
            self.show_warning("Not found", "No record with that ID.")
            return
        self._update_display(lug)

    def delete_luggage(self):
        id_ = self.search_id.text().strip()
        if not id_:
            return
        ok = self.db.delete(id_)
        if ok:
            self.show_info("Deleted", "Record deleted.")
            self._update_display(None)
        else:
            self.show_warning("Nothing to delete", "Record not found.")

    def save_current_qr(self):
        if not hasattr(self, "_current_lug") or self._current_lug is None:
            self.show_warning("No QR", "There is no current luggage QR to save.")
            return
        src = Path(self._current_lug.qr_path)
        if not src.exists():
            self.show_warning("Missing", "QR not found on disk.")
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save QR", src.name, "PNG Files (*.png)")
        if dest:
            Path(dest).write_bytes(src.read_bytes())
            self.show_info("Saved", f"Saved to:<br>{dest}")

    def scan_camera(self):
        dlg = CameraScanner(self)

        def on_scanned(payload: dict):
            lid = payload.get("id", "")
            if not lid:
                self.show_warning("QR", "QR does not contain a luggage ID.")
                return
            lug = self.db.get(lid)
            if lug:
                self._update_display(lug)
            else:
                self.show_info("QR Details", f"<pre>{json.dumps(payload, indent=2)}</pre>")

        dlg.scanned.connect(on_scanned)
        dlg.start()

    def _update_display(self, lug: Luggage | None):
        self._current_lug = lug
        if lug is None:
            self.qr_label.clear()
            self.v_flight.setText("N/A"); self.v_area.setText("N/A"); self.v_belt.setText("N/A"); self.v_id.setText("N/A")
            return

        self.v_flight.setText(lug.flight_no)
        self.v_area.setText(lug.belt_area)
        self.v_belt.setText(lug.belt_no)
        self.v_id.setText(lug.id)

        pm = QtGui.QPixmap()
        if pm.load(lug.qr_path):
            self.qr_label.setPixmap(pm.scaled(
                300, 300,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.qr_label.clear()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    w = Main()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
