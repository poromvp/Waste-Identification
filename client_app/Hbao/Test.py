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

        # ===== CẤU HINH FONT =====
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

        # Định nghĩa các nhóm rác thải dựa trên file data.yaml
        # 0: bottle, 1: straw, 2: bag, 3: HDPE (Rác thải nhựa: 0-3)
        # 4: glass, 5: card_board, 6: metal, 7: pin, 8: paper (Rác thải rắn: 4-8)
        # 9: nylon (Nylon: 9)
        # 10: bang_gat, 11: glove, 12: kim_tiem, 13: mask (Rác thải y tế: 10-13)
        self.waste_groups = {
            "Rác thải nhựa ♻️ (0-3)": [0, 1, 2, 3],
            "Rác thải rắn 🧱 (4-8)": [4, 5, 6, 7, 8],
            "Nylon 🛍️ (9)": [9],
            "Rác thải y tế 💉 (10-13)": [10, 11, 12, 13]
        }
        self.initial_camera_text = "Camera chưa bật"
        self.initial_result_text = "Chưa có dữ liệu nhận diện."

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

        self.camera_label = tk.Label(self.camera_frame, bg="#ecf0f1", text=self.initial_camera_text,
                                     font=("Segoe UI", 12), fg="#7f8c8d")
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

        self.history_button = ttk.Button(btn_frame, text="Xem Lịch Sử", style="Orange.TButton",
                                         command=self.open_history)
        self.history_button.pack(fill=tk.X, pady=8)

        # ===== VÙNG KẾT QUẢ NHẬN DIỆN =====
        result_frame = tk.LabelFrame(right_panel, text=" Kết quả nhận diện ", font=self.subtitle_font, bg="white",
                                     fg="#2c3e50", bd=1, relief=tk.SOLID, padx=15, pady=15)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=20)

        self.result_text = tk.Text(result_frame, height=10, font=("Consolas", 10), bg="#f8f9fa", fg="#2c3e50", bd=0,
                                   relief=tk.FLAT, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, self.initial_result_text)
        self.result_text.config(state=tk.DISABLED)

        # ===== THANH TRẠNG THÁI =====
        status_frame = tk.Frame(self.root, bg="#34495e", height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text=self.status_text, font=("Segoe UI", 10), bg="#34495e",
                                     fg="#bdc3c7", anchor="w")
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

    def format_results(self, results):
        """Phân loại và định dạng kết quả nhận diện theo 4 nhóm rác thải (không đếm số lượng)."""

        # 1. Lấy class ID (int) từ kết quả YOLO
        all_class_ids = [int(box.cls[0]) for r in results for box in r.boxes]
        unique_class_ids = set(all_class_ids)  # Chỉ lấy các class_id duy nhất

        # 2. Gom nhóm theo loại rác thải
        grouped_results = {}
        for group_name, group_ids in self.waste_groups.items():
            detected_labels_in_group = []  # Dùng list để giữ thứ tự
            for class_id in group_ids:
                if class_id in unique_class_ids:  # Kiểm tra xem class_id này có được phát hiện không
                    label = self.model.names.get(class_id, f"Class {class_id}")
                    if label not in detected_labels_in_group:  # Tránh trùng lặp tên nhãn
                        detected_labels_in_group.append(label)

            if detected_labels_in_group:
                # YÊU CẦU MỚI: Chỉ thêm tên nhãn, không thêm số lượng
                grouped_results[group_name] = [f"• {label}" for label in detected_labels_in_group]

        # 3. Định dạng thành chuỗi hiển thị
        total_detections = len(all_class_ids)

        # Nếu không phát hiện vật thể HOẶC không có vật thể nào thuộc nhóm
        if total_detections == 0 or not grouped_results:
            return "Không phát hiện rác thải."

        # YÊU CẦU MỚI: Bỏ dòng đếm tổng số
        display_text = ""

        for group_name, items in grouped_results.items():
            display_text += f"\n--- {group_name} ---\n"
            display_text += "\n".join(items) + "\n"

        # Trả về chuỗi đã loại bỏ khoảng trắng thừa ở đầu/cuối
        return display_text.strip()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        try:
            # Điều chỉnh kích thước khung hình cho phù hợp với khu vực hiển thị
            h, w = frame.shape[:2]
            target_w = 780
            target_h = 480
            # Tính tỉ lệ để giữ tỉ lệ khung hình nếu cần. Ở đây dùng fixed size (780, 480) như code gốc
            frame_processed = cv2.resize(frame, (target_w, target_h))

            results = self.model(frame_processed, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            # --- Cải tiến: Phân loại kết quả nhận diện ---
            formatted_text = self.format_results(results)

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, formatted_text)
            self.result_text.config(state=tk.DISABLED)

            # Cập nhật trạng thái thanh status
            if "Không phát hiện" in formatted_text:
                self.update_status("Không phát hiện vật thể", "#95a5a6")
            else:
                detected_groups = [name.split()[0] for name in self.waste_groups.keys() if name in formatted_text]
                if detected_groups:
                    self.update_status(f"Phát hiện nhóm: {', '.join(detected_groups)}", "#1abc9c")
                else:
                    self.update_status("Đã phát hiện vật thể", "#1abc9c")

            # Hiển thị khung hình
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk, text="")

        except Exception as e:
            print(f"Lỗi camera: {e}")

        self.root.after(30, self.update_frame)

    def capture_image(self):
        """Chụp ảnh → Lưu ảnh + file .txt → Tắt camera → Hiển thị ảnh + kết quả"""
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

        # --- Phân loại kết quả nhận diện ---
        formatted_text = self.format_results(results)

        # Lưu ảnh
        output_path = self.get_next_filename()
        cv2.imwrite(output_path, annotated_frame)

        # --- YÊU CẦU MỚI: Lưu thông tin phân loại vào file .txt ---
        try:
            txt_path = os.path.splitext(output_path)[0] + ".txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
        except Exception as e:
            print(f"Lỗi lưu file txt: {e}")
        # --- KẾT THÚC THAY ĐỔI ---

        # Cập nhật giao diện ảnh
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk, text="")

        # Cập nhật vùng kết quả
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, formatted_text)
        self.result_text.config(state=tk.DISABLED)

        # Cập nhật trạng thái
        filename = os.path.basename(output_path)
        self.update_status(f"Đã chụp & lưu: {filename}", "#1abc9c")

        # Tắt camera
        self.stop_camera(revert_label=False)  # Giữ lại ảnh chụp trên camera_label

    def upload_image(self):
        """Tải ảnh → Xử lý → Hiển thị + lưu ảnh + lưu file .txt"""
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

            # --- Phân loại kết quả nhận diện ---
            formatted_text = self.format_results(results)

            # Lưu ảnh
            output_path = self.get_next_filename()
            cv2.imwrite(output_path, annotated_frame)

            # --- YÊU CẦU MỚI: Lưu thông tin phân loại vào file .txt ---
            try:
                txt_path = os.path.splitext(output_path)[0] + ".txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)
            except Exception as e:
                print(f"Lỗi lưu file txt: {e}")
            # --- KẾT THÚC THAY ĐỔI ---

            # Hiển thị ảnh
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk, text="")

            # Cập nhật kết quả
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, formatted_text)
            self.result_text.config(state=tk.DISABLED)

            self.update_status(f"Đã xử lý & lưu: {os.path.basename(output_path)}", "#1abc9c")

        except Exception as e:
            self.update_status(f"Lỗi xử lý ảnh: {e}", "#e74c3c")

    def stop_camera(self, revert_label=True):
        """Dừng camera và khôi phục giao diện nếu revert_label=True (tắt camera).
        Nếu revert_label=False (chụp ảnh), giữ lại ảnh trên camera_label."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_button.config(text="Bật Camera")
        self.upload_button.config(state=tk.NORMAL)
        self.capture_button.pack_forget()

        # --- Cải tiến: Khôi phục nền cũ khi tắt camera ---
        if revert_label:
            # Đảm bảo camera_label không giữ tham chiếu đến ảnh cũ
            self.camera_label.imgtk = None
            self.camera_label.config(image='', text=self.initial_camera_text, fg="#7f8c8d")

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, self.initial_result_text)
            self.result_text.config(state=tk.DISABLED)

            self.update_status(self.status_text)  # Khôi phục status sau khi load model

    # ===== [START] HÀM ĐÃ THAY ĐỔI (Thêm canh giữa) =====
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
        # Kích thước cố định cho cửa sổ lịch sử
        window_width = 900
        window_height = 600

        self.history_window.configure(bg="#f4f6f9")
        self.history_window.protocol("WM_DELETE_WINDOW", self.on_history_close)

        # ===== YÊU CẦU MỚI: Canh giữa cửa sổ =====
        # Lấy kích thước và vị trí cửa sổ chính
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()

        # Tính toán vị trí x, y để canh giữa
        center_x = main_x + (main_width // 2) - (window_width // 2)
        center_y = main_y + (main_height // 2) - (window_height // 2)

        # Áp dụng vị trí và kích thước
        self.history_window.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.history_window.resizable(False, False)  # Cố định kích thước
        # ===== KẾT THÚC THAY ĐỔI =====

        tk.Label(
            self.history_window,
            text="Các ảnh đã phát hiện (Nhấn để xem chi tiết)",  # Hướng dẫn
            font=("Segoe UI", 16, "bold"),
            bg="#f4f6f9",
            fg="#2c3e50"
        ).pack(pady=15)

        canvas = tk.Canvas(self.history_window, bg="#f4f6f9", highlightthickness=0)
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
            key=lambda x: os.path.getctime(os.path.join(self.detection_dir, x)),
            reverse=True  # Hiển thị ảnh mới nhất trước
        )

        cols = 4
        for idx, filename in enumerate(image_files):
            path = os.path.join(self.detection_dir, filename)
            txt_path = os.path.splitext(path)[0] + ".txt"
            try:
                img = Image.open(path)
                img.thumbnail((180, 180))
                photo = ImageTk.PhotoImage(img)

                frame = tk.Frame(scrollable_frame, bg="white", relief=tk.SOLID, bd=1)
                frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10)

                lbl_img = tk.Label(frame, image=photo, bg="white", cursor="hand2")
                lbl_img.image = photo
                lbl_img.pack()

                info = f"{filename}\n{datetime.fromtimestamp(os.path.getctime(path)).strftime('%H:%M %d/%m/%Y')}"
                lbl_info = tk.Label(frame, text=info, font=("Segoe UI", 9), bg="white", fg="#34495e", justify=tk.CENTER)
                lbl_info.pack(pady=5)

                lbl_img.bind("<Button-1>", lambda e, p=path, t=txt_path: self.show_history_detail(p, t))

            except Exception as e:
                print(f"Không load được ảnh: {e}")

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_window.transient(self.root)
        self.history_window.grab_set()

    # ===== [END] HÀM ĐÃ THAY ĐỔI =====

    def on_history_close(self):
        self.history_window.grab_release()
        self.history_window.destroy()
        self.history_window = None

    # ===== [START] HÀM ĐÃ THAY ĐỔI (Thêm canh giữa) =====
    def show_history_detail(self, image_path, txt_path):
        """Hiển thị chi tiết ảnh và kết quả phân loại."""

        detail_window = tk.Toplevel(self.root)
        detail_window.title(os.path.basename(image_path))
        detail_window.configure(bg="#f4f6f9")

        # --- Frame chính ---
        main_frame = tk.Frame(detail_window, bg="#f4f6f9")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # --- 1. Hiển thị ảnh (bên trái) ---
        img_frame = tk.Frame(main_frame, bg="white", bd=1, relief=tk.SOLID)
        img_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))

        try:
            img = Image.open(image_path)

            # Giới hạn kích thước ảnh xem full
            max_w = self.root.winfo_screenwidth() * 0.6
            max_h = self.root.winfo_screenheight() * 0.7

            current_w, current_h = img.size

            if current_w > max_w or current_h > max_h:
                ratio = min(max_w / current_w, max_h / current_h)
                new_w = int(current_w * ratio)
                new_h = int(current_h * ratio)
                img = img.resize((new_w, new_h),
                                 Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            lbl_img = tk.Label(img_frame, image=photo, bg="white")
            lbl_img.image = photo
            lbl_img.pack(padx=10, pady=10)
        except Exception as e:
            tk.Label(img_frame, text=f"Lỗi tải ảnh: {e}", bg="white").pack(padx=10, pady=10)

        # --- 2. Hiển thị kết quả (bên phải) ---
        result_frame = tk.LabelFrame(main_frame, text=" Kết quả phân loại ", font=self.subtitle_font, bg="white",
                                     fg="#2c3e50", bd=1, relief=tk.SOLID, padx=15, pady=15)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Đọc thông tin từ file .txt
        info_text = "Không có thông tin phân loại."
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                info_text = f.read()

        if not info_text.strip():
            info_text = "Không phát hiện rác thải."

        # Hiển thị text
        txt_display = tk.Text(result_frame, font=("Consolas", 10), bg="#f8f9fa", fg="#2c3e50", bd=0,
                              relief=tk.FLAT, wrap=tk.WORD, width=40, height=20)
        txt_display.pack(fill=tk.BOTH, expand=True)
        txt_display.insert(tk.END, info_text)
        txt_display.config(state=tk.DISABLED)

        # ===== YÊU CẦU MỚI: Canh giữa cửa sổ =====
        # Phải update để tkinter tính toán kích thước thực tế của cửa sổ
        detail_window.update_idletasks()

        # Lấy kích thước cửa sổ chi tiết SAU KHI đã thêm nội dung
        popup_width = detail_window.winfo_width()
        popup_height = detail_window.winfo_height()

        # Lấy kích thước và vị trí cửa sổ chính
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()

        # Tính toán vị trí x, y để canh giữa
        center_x = main_x + (main_width // 2) - (popup_width // 2)
        center_y = main_y + (main_height // 2) - (popup_height // 2)

        detail_window.geometry(f"{popup_width}x{popup_height}+{center_x}+{center_y}")
        detail_window.resizable(False, False)  # Cố định kích thước
        # ===== KẾT THÚC THAY ĐỔI =====

        detail_window.transient(self.root)
        detail_window.grab_set()

    # ===== [END] HÀM ĐÃ THAY ĐỔI =====

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