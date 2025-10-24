# giaodienmoi.py (phiên bản: CHỤP ẢNH → TẮT CAMERA → HIỂN THỊ NGAY)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import numpy as np
from datetime import datetime


class WasteRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Garbage Identify")
        self.root.state('zoomed')
        self.root.configure(bg="#f4f6f9")

        # ===== CẤU HÌNH FONT =====
        self.title_font = tkfont.Font(family="Segoe UI", size=24, weight="bold")
        self.subtitle_font = tkfont.Font(family="Segoe UI", size=12)
        self.button_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")

        # ===== BIẾN HỆ THỐNG =====
        self.cap = None
        self.running = False
        self.model = None
        self.detection_count = 0
        self.detection_dir = "detections"
        self.history_window = None  # Cho lịch sử

        # Tạo thư mục lưu ảnh
        if not os.path.exists(self.detection_dir):
            os.makedirs(self.detection_dir)

        # ===== TẢI MODEL =====
        self.load_model()

        # ===== GIAO DIỆN CHÍNH =====
        self.setup_ui()

    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "ModelAI.pt")
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.status_text = "Sẵn sàng: Model YOLOv8 đã tải thành công"
            except Exception as e:
                self.model = None
                self.status_text = f"Lỗi tải model: {e}"
        else:
            self.model = None
            self.status_text = "Cảnh báo: Không tìm thấy file ModelAI.pt"

    def setup_ui(self):
        # ===== HEADER =====
        header = tk.Frame(self.root, bg="#2c3e50", height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        header.grid_rowconfigure(0, weight=1)
        header.grid_rowconfigure(1, weight=1)
        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="Garbage Identify",
            font=("Segoe UI", 26, "bold"),
            bg="#2c3e50",
            fg="#1abc9c",
            anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=25, pady=(12, 0))

        tk.Label(
            header,
            text="Nhận diện & Phân loại Rác thải bằng AI",
            font=("Segoe UI", 11),
            bg="#2c3e50",
            fg="#bdc3c7",
            anchor="w"
        ).grid(row=1, column=0, sticky="w", padx=25, pady=(0, 12))

        # ===== MAIN CONTAINER =====
        main_container = tk.Frame(self.root, bg="#f4f6f9")
        main_container.pack(expand=True, fill=tk.BOTH, padx=25, pady=20)

        # ===== LEFT PANEL: CAMERA =====
        left_panel = tk.Frame(main_container, bg="white", relief=tk.SOLID, bd=1)
        left_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 15))

        cam_title = tk.Label(
            left_panel,
            text="Camera Trực Tiếp",
            font=self.subtitle_font,
            bg="white",
            fg="#34495e",
            anchor="w"
        )
        cam_title.pack(anchor="w", padx=15, pady=(15, 5))

        self.camera_frame = tk.Frame(left_panel, bg="#ecf0f1", bd=2, relief=tk.SUNKEN)
        self.camera_frame.pack(expand=True, fill=tk.BOTH, padx=15, pady=(0, 15))

        self.camera_label = tk.Label(self.camera_frame, bg="#ecf0f1", text="Camera chưa bật", font=("Segoe UI", 12), fg="#7f8c8d")
        self.camera_label.pack(expand=True)

        # ===== RIGHT PANEL: KẾT QUẢ & NÚT =====
        right_panel = tk.Frame(main_container, bg="#f4f6f9", width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        right_panel.pack_propagate(False)

        tk.Label(
            right_panel,
            text="Điều khiển & Kết quả",
            font=self.subtitle_font,
            bg="#f4f6f9",
            fg="#2c3e50",
            anchor="w"
        ).pack(anchor="w", pady=(0, 15))

        # ===== NÚT ĐIỀU KHIỂN =====
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("Green.TButton", background="#1abc9c", foreground="white", font=self.button_font, padding=12)
        style.map("Green.TButton", background=[('active', '#16a085')])

        style.configure("Blue.TButton", background="#3498db", foreground="white", font=self.button_font, padding=12)
        style.map("Blue.TButton", background=[('active', '#2980b9')])

        style.configure("Gray.TButton", background="#95a5a6", foreground="white", font=self.button_font, padding=12)
        style.map("Gray.TButton", background=[('active', '#7f8c8d')])

        style.configure("Orange.TButton", background="#e67e22", foreground="white", font=self.button_font, padding=12)
        style.map("Orange.TButton", background=[('active', '#d35400')])

        btn_frame = tk.Frame(right_panel, bg="#f4f6f9")
        btn_frame.pack(pady=10, fill=tk.X, padx=15)

        self.start_button = ttk.Button(btn_frame, text="Bật Camera", style="Green.TButton", command=self.toggle_camera)
        self.start_button.pack(fill=tk.X, pady=8)

        self.upload_button = ttk.Button(btn_frame, text="Tải Ảnh Lên", style="Blue.TButton", command=self.upload_image)
        self.upload_button.pack(fill=tk.X, pady=8)

        self.capture_button = ttk.Button(btn_frame, text="Chụp Ảnh", style="Gray.TButton", command=self.capture_image)
        self.capture_button.pack(fill=tk.X, pady=8)
        self.capture_button.pack_forget()

        self.history_button = ttk.Button(btn_frame, text="Xem Lịch Sử", style="Orange.TButton", command=self.open_history)
        self.history_button.pack(fill=tk.X, pady=8)

        # ===== VÙNG KẾT QUẢ NHẬN DIỆN =====
        result_frame = tk.LabelFrame(right_panel, text=" Kết quả nhận diện ", font=self.subtitle_font, bg="white", fg="#2c3e50", bd=1, relief=tk.SOLID, padx=15, pady=15)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=20)

        self.result_text = tk.Text(result_frame, height=10, font=("Consolas", 10), bg="#f8f9fa", fg="#2c3e50", bd=0, relief=tk.FLAT, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "Chưa có dữ liệu nhận diện.")
        self.result_text.config(state=tk.DISABLED)

        # ===== THANH TRẠNG THÁI =====
        status_frame = tk.Frame(self.root, bg="#34495e", height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text=self.status_text, font=("Segoe UI", 10), bg="#34495e", fg="#bdc3c7", anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)

        self.update_status(self.status_text)

    def update_status(self, text, color="#bdc3c7"):
        self.status_label.config(text=text, fg=color)
        self.root.update_idletasks()

    def get_next_filename(self):
        """Tạo tên file theo thứ tự: detect_001.jpg"""
        self.detection_count += 1
        return os.path.join(self.detection_dir, f"detect_{self.detection_count:03d}.jpg")

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if not self.model:
            self.update_status("Lỗi: Model chưa tải!", "#e74c3c")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.update_status("Không thể mở camera", "#e74c3c")
            return

        self.running = True
        self.start_button.config(text="Tắt Camera")
        self.upload_button.config(state=tk.DISABLED)
        self.capture_button.pack(fill=tk.X, pady=8)
        self.update_status("Camera đang chạy...", "#1abc9c")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Đang quét camera...\n")
        self.result_text.config(state=tk.DISABLED)
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        try:
            frame = cv2.resize(frame, (780, 480))
            results = self.model(frame, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            labels = [self.model.names[int(box.cls[0])] for r in results for box in r.boxes]
            waste_counts = {}
            for label in labels:
                waste_counts[label] = waste_counts.get(label, 0) + 1

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            if waste_counts:
                for label, count in waste_counts.items():
                    self.result_text.insert(tk.END, f"• {label}: {count} vật thể\n")
                self.update_status(f"Phát hiện: {', '.join(waste_counts.keys())}", "#1abc9c")
            else:
                self.result_text.insert(tk.END, "Không phát hiện rác thải.\n")
                self.update_status("Không phát hiện vật thể", "#95a5a6")
            self.result_text.config(state=tk.DISABLED)

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk, text="")

        except Exception as e:
            print(f"Lỗi camera: {e}")

        self.root.after(30, self.update_frame)

    def capture_image(self):
        """Chụp ảnh → Lưu → Tắt camera → Hiển thị ảnh + kết quả trên giao diện"""
        if not self.cap or not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.update_status("Lỗi chụp ảnh", "#e74c3c")
            return

        # Resize để xử lý
        frame_resized = cv2.resize(frame, (780, 480))
        results = self.model(frame_resized, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        # Lấy nhãn và đếm
        labels = [self.model.names[int(box.cls[0])] for r in results for box in r.boxes]
        waste_counts = {}
        for label in labels:
            waste_counts[label] = waste_counts.get(label, 0) + 1

        waste_info_lines = [f"{label}: {count} vật thể" for label, count in waste_counts.items()]

        # Lưu ảnh
        output_path = self.get_next_filename()
        cv2.imwrite(output_path, annotated_frame)

        # Cập nhật giao diện
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk, text="")

        # Cập nhật vùng kết quả
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        if waste_info_lines:
            for line in waste_info_lines:
                self.result_text.insert(tk.END, f"• {line}\n")
        else:
            self.result_text.insert(tk.END, "Không phát hiện rác thải.\n")
        self.result_text.config(state=tk.DISABLED)

        # Cập nhật trạng thái
        filename = os.path.basename(output_path)
        self.update_status(f"Đã chụp & lưu: {filename}", "#1abc9c")

        # Tắt camera
        self.stop_camera()

    def upload_image(self):
        """Tải ảnh → Xử lý → Hiển thị + lưu"""
        if self.running:
            self.stop_camera()

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh rác thải",
            filetypes=[("Hình ảnh", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        try:
            frame = cv2.imread(file_path)
            if frame is None:
                self.update_status("Không đọc được ảnh", "#e74c3c")
                return

            frame_resized = cv2.resize(frame, (780, 480))
            results = self.model(frame_resized, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            labels = [self.model.names[int(box.cls[0])] for r in results for box in r.boxes]
            waste_counts = {}
            for label in labels:
                waste_counts[label] = waste_counts.get(label, 0) + 1

            waste_info_lines = [f"{label}: {count} vật thể" for label, count in waste_counts.items()]

            # Lưu ảnh
            output_path = self.get_next_filename()
            cv2.imwrite(output_path, annotated_frame)

            # Hiển thị ảnh
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk, text="")

            # Cập nhật kết quả
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            if waste_info_lines:
                for line in waste_info_lines:
                    self.result_text.insert(tk.END, f"• {line}\n")
            else:
                self.result_text.insert(tk.END, "Không phát hiện rác thải.\n")
            self.result_text.config(state=tk.DISABLED)

            self.update_status(f"Đã xử lý & lưu: {os.path.basename(output_path)}", "#1abc9c")

        except Exception as e:
            self.update_status(f"Lỗi xử lý ảnh: {e}", "#e74c3c")

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_button.config(text="Bật Camera")
        self.upload_button.config(state=tk.NORMAL)
        self.capture_button.pack_forget()
        # Không thay đổi camera_label nếu đang hiển thị ảnh chụp

    def open_history(self):
        """Mở cửa sổ xem lại ảnh (chỉ 1 lần)"""
        if hasattr(self, 'history_window') and self.history_window and self.history_window.winfo_exists():
            self.history_window.lift()
            return

        if not os.path.exists(self.detection_dir) or not os.listdir(self.detection_dir):
            messagebox.showinfo("Lịch sử", "Chưa có ảnh nào được lưu.")
            return

        self.history_window = tk.Toplevel(self.root)
        self.history_window.title("Lịch Sử Phát Hiện")
        self.history_window.geometry("900x600")
        self.history_window.configure(bg="#f4f6f9")
        self.history_window.protocol("WM_DELETE_WINDOW", self.on_history_close)

        tk.Label(
            self.history_window,
            text="Các ảnh đã phát hiện",
            font=("Segoe UI", 16, "bold"),
            bg="#f4f6f9",
            fg="#2c3e50"
        ).pack(pady=15)

        canvas = tk.Canvas(self.history_window, bg="#f4f6f9")
        scrollbar = ttk.Scrollbar(self.history_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f4f6f9")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        image_files = sorted(
            [f for f in os.listdir(self.detection_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: os.path.getctime(os.path.join(self.detection_dir, x))
        )

        cols = 4
        for idx, filename in enumerate(image_files):
            path = os.path.join(self.detection_dir, filename)
            try:
                img = Image.open(path)
                img.thumbnail((180, 180))
                photo = ImageTk.PhotoImage(img)

                frame = tk.Frame(scrollable_frame, bg="white", relief=tk.SOLID, bd=1)
                frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10)

                lbl_img = tk.Label(frame, image=photo, bg="white")
                lbl_img.image = photo
                lbl_img.pack()

                info = f"{filename}\n{datetime.fromtimestamp(os.path.getctime(path)).strftime('%H:%M %d/%m/%Y')}"
                lbl_info = tk.Label(frame, text=info, font=("Segoe UI", 9), bg="white", fg="#34495e", justify=tk.CENTER)
                lbl_info.pack(pady=5)

                lbl_img.bind("<Button-1>", lambda e, p=path: self.show_full_image(p))

            except Exception as e:
                print(f"Không load được ảnh: {e}")

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def on_history_close(self):
        self.history_window.destroy()
        self.history_window = None

    def show_full_image(self, path):
        img_window = tk.Toplevel(self.root)
        img_window.title(os.path.basename(path))
        img = Image.open(path)
        img = img.resize((800, 600), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(img_window, image=photo)
        lbl.image = photo
        lbl.pack()

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