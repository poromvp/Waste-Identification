import tkinter as tk
from tkinter import ttk
from tkinter import filedialog  # <<< THÊM VÀO
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import numpy as np
import time
from PanelThongBao import WasteInfoDialog


class WasteRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm Nhận diện Rác thải")
        self.root.geometry("900x650")  # Giữ kích thước mặc định nếu người dùng thu nhỏ
        self.root.state('zoomed')      # <<< THÊM VÀO: Mở full màn hình (maximized)
        self.root.configure(bg="#f0f0f0")

        # ===== Tiêu đề =====
        self.title_label = tk.Label(
            self.root,
            text="Nhận Diện Rác Thải",
            font=("Arial", 22, "bold"),
            bg="#f0f0f0",
            fg="#333",
        )
        self.title_label.pack(pady=15)

        # ===== Khung camera (không dùng relief để tránh nhấp nháy) =====
        self.camera_frame = tk.Frame(
            self.root, bg="#fff", highlightbackground="#aaa", highlightthickness=2
        )
        self.camera_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        self.camera_label = tk.Label(self.camera_frame, bg="#fff")
        self.camera_label.pack(expand=True, fill=tk.BOTH)

        # ===== Nút điều khiển =====
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)

        self.button_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.button_frame.pack(side=tk.BOTTOM, pady=15)

        self.start_button = ttk.Button(
            self.button_frame, text="Bật Camera", command=self.toggle_camera  # <<< THAY ĐỔI
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        # <<< THÊM NÚT TẢI ẢNH >>>
        self.upload_button = ttk.Button(
            self.button_frame, text="Tải Ảnh Lên", command=self.upload_image
        )
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.capture_button = ttk.Button(
            self.button_frame, text="Chụp Ảnh", command=self.capture_image
        )
        self.capture_button.pack_forget()

        # ===== Trạng thái =====
        self.status_label = tk.Label(
            self.root, text="", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#666"
        )
        self.status_label.pack(side=tk.BOTTOM, pady=10)

        # ===== Biến hệ thống =====
        self.cap = None
        self.running = False
        self.photo = None
        self.frame_count = 0

        # ===== Biến lưu khung nhận dạng trước đó để tránh nhấp nháy =====
        self.last_boxes = []
        self.last_confidences = []
        self.last_labels = []
        self.last_update_time = 0

        # ===== Load model YOLO =====
        model_path = "D:/CNTT-SGU/HK5/Python/TrainAIFinal/TrainAIFinal/train/yolo_train2/weights/best.pt"

        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.status_label.config(text="Model YOLOv8 đã được load thành công")
        else:
            self.model = None
            self.status_label.config(text="Không tìm thấy model YOLOv8")

    # ==================== ĐIỀU KHIỂN CHUNG ====================

    # <<< THÊM HÀM MỚI >>>
    def toggle_camera(self):
        """Bật hoặc tắt camera."""
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    # <<< THÊM HÀM MỚI >>>
    def upload_image(self):
        """Mở file dialog để chọn ảnh và xử lý."""
        # Đảm bảo camera tắt
        if self.running:
            self.stop_camera()

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return  # Người dùng hủy

        try:
            frame = cv2.imread(file_path)
            if frame is None:
                self.status_label.config(text="Không thể đọc file ảnh")
                return

            # Resize cho vừa khung hiển thị
            frame_resized = cv2.resize(frame, (800, 500))

            # Chạy YOLO
            results = self.model(frame_resized, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            # Lấy danh sách loại rác và đếm
            labels = [self.model.names[int(box.cls[0])] for r in results for box in r.boxes]
            waste_counts = {}
            for label in labels:
                waste_counts[label] = waste_counts.get(label, 0) + 1

            if waste_counts:
                waste_info_lines = [f"{label}: {count} vật thể" for label, count in waste_counts.items()]
                waste_info = "\n".join(waste_info_lines)
                waste_types = list(waste_counts.keys())
            else:
                waste_info = "Không phát hiện vật thể nào."
                waste_types = []

            # Lưu ảnh detection
            output_path = "detected_waste.jpg"
            cv2.imwrite(output_path, annotated_frame)

            # Hiển thị ảnh đã nhận diện
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

            self.status_label.config(text=f"Đã xử lý ảnh: {os.path.basename(file_path)}")

            # Mở hộp thoại thông tin
            WasteInfoDialog(self.root, waste_types=waste_types, waste_info=waste_info)

        except Exception as e:
            self.status_label.config(text=f"Lỗi khi xử lý ảnh: {e}")

    # ==================== CAMERA ====================
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.config(text="Không thể mở camera")
                self.cap = None  # Đặt lại cap để có thể thử lại
                return

        self.running = True
        # <<< THAY ĐỔI >>>
        self.start_button.config(text="Tắt Camera")
        self.upload_button.config(state=tk.DISABLED)  # Vô hiệu hóa nút tải ảnh

        self.capture_button.pack(side=tk.LEFT, padx=10)
        self.status_label.config(text="Camera đang chạy...")
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Nếu không đọc được frame, thử lại sau
            self.root.after(30, self.update_frame)
            return

        try:
            # Resize cho vừa khung hiển thị
            frame = cv2.resize(frame, (800, 500))

            # Dự đoán và YOLO tự vẽ khung
            results = self.model(frame, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            # Lấy nhãn (label)
            labels = [self.model.names[int(box.cls[0])] for r in results for box in r.boxes]

            # Cập nhật label trạng thái
            if labels:
                unique_labels = sorted(list(set(labels)))
                self.status_label.config(text="Phát hiện: " + ", ".join(unique_labels))
            else:
                self.status_label.config(text="Không phát hiện vật thể nào")

            # Hiển thị khung ảnh trong Tkinter
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        except Exception as e:
            print(f"Lỗi trong update_frame: {e}")  # Ghi log lỗi
            # Có thể dừng camera nếu lỗi nghiêm trọng
            # self.stop_camera()
            # self.status_label.config(text=f"Lỗi: {e}")

        # Lặp lại mỗi 30ms
        self.root.after(30, self.update_frame)

    def capture_image(self):
        if self.cap and self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="Lỗi khi chụp ảnh")
                return

            # Chạy YOLO để vẽ box trên ảnh chụp
            results = self.model(frame, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()  # ảnh có khung sẵn

            # <<< THAY ĐỔI LOGIC LẤY THÔNG TIN >>>
            # Lấy danh sách loại rác và đếm
            labels = [self.model.names[int(box.cls[0])] for r in results for box in r.boxes]
            waste_counts = {}
            for label in labels:
                waste_counts[label] = waste_counts.get(label, 0) + 1

            if waste_counts:
                waste_info_lines = [f"{label}: {count} vật thể" for label, count in waste_counts.items()]
                waste_info = "\n".join(waste_info_lines)
                waste_types = list(waste_counts.keys())
            else:
                waste_info = "Không phát hiện vật thể nào."
                waste_types = []

            # Lưu ảnh detection
            output_path = "detected_waste.jpg"
            cv2.imwrite(output_path, annotated_frame)
            self.status_label.config(text="Ảnh đã được lưu! Đang tắt camera...")

            # Mở hộp thoại thông tin
            WasteInfoDialog(self.root, waste_types=waste_types, waste_info=waste_info)

            # <<< THÊM VÀO: Tắt camera sau khi chụp >>>
            self.stop_camera()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_label.config(image="", text="Camera đã tắt")

        # <<< THAY ĐỔI >>>
        self.start_button.config(text="Bật Camera")
        self.upload_button.config(state=tk.NORMAL)  # Kích hoạt lại nút tải ảnh

        self.capture_button.pack_forget()
        self.status_label.config(text="Camera đã dừng")

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = WasteRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()