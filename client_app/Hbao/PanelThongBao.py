import tkinter as tk
from tkinter import ttk
import os


class WasteInfoDialog(tk.Toplevel):
    def __init__(self, parent, waste_types=None, waste_info="Không có thông tin"):
        super().__init__(parent)
        self.title("Thông tin Rác thải")
        self.geometry("700x400")
        self.configure(bg="#f0f0f0")
        self.resizable(False, False)
        self.transient(parent)

        # <<< THÊM VÀO: Tạo lớp phủ mờ (dim overlay) >>>
        self.overlay = tk.Toplevel(parent)
        self.overlay.overrideredirect(True)  # Bỏ viền và thanh tiêu đề
        self.overlay.config(bg='black')  # Màu đen

        # Lấy kích thước màn hình
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Đặt lớp phủ che toàn màn hình
        self.overlay.geometry(f'{screen_width}x{screen_height}+0+0')
        self.overlay.attributes('-alpha', 0.7)  # Độ mờ 70%
        self.overlay.lift()  # Nâng lớp phủ lên trên cửa sổ chính

        # <<< THÊM VÀO: Canh giữa màn hình >>>
        self.update_idletasks()  # Cập nhật để lấy kích thước dialog chính xác
        width = self.winfo_width()  # 700
        height = self.winfo_height()  # 400

        # Tính toán vị trí x, y
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        # Đặt lại vị trí cho dialog
        self.geometry(f'{width}x{height}+{x}+{y}')

        self.lift()  # Nâng dialog lên trên lớp phủ
        self.grab_set()  # Đã có: Đảm bảo modal (không tương tác được với cửa sổ sau)

        # Tiêu đề
        self.title_label = tk.Label(
            self,
            text="Kết quả Nhận diện",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#333"
        )
        self.title_label.pack(pady=15)

        # Khung thông tin rác thải
        self.info_frame = tk.Frame(self, bg="#fff", bd=2, relief=tk.SUNKEN)
        self.info_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Nhãn danh sách loại rác thải
        waste_types_text = ", ".join(waste_types) if waste_types else "Không xác định"
        self.waste_type_label = tk.Label(
            self.info_frame,
            text=f"Loại rác thải: {waste_types_text}",
            font=("Arial", 12, "bold"),
            bg="#fff",
            fg="blue",
            wraplength=650,  # <<< THAY ĐỔI: Tăng wraplength
            justify="left"
        )
        self.waste_type_label.pack(pady=10, anchor="w", padx=20)

        # Nhãn thông tin chi tiết
        self.waste_info_label = tk.Label(
            self.info_frame,
            text=waste_info,
            font=("Arial", 12),
            bg="#fff",
            fg="#000",
            wraplength=650,  # <<< THAY ĐỔI: Tăng wraplength
            justify="left"
        )
        self.waste_info_label.pack(pady=10, anchor="w", padx=20)

        # Nút xem ảnh detection
        if os.path.exists("detected_waste.jpg"):
            view_btn = ttk.Button(
                self.info_frame,
                text="Xem ảnh detection",
                command=lambda: os.startfile("detected_waste.jpg"),
                style="TButton"
            )
            view_btn.pack(pady=5)
        else:
            no_image_label = tk.Label(
                self.info_frame,
                text="Không có ảnh detection",
                font=("Arial", 10, "italic"),
                bg="#fff",
                fg="#666"
            )
            no_image_label.pack(pady=5)

        # Nút đóng
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)
        self.close_button = ttk.Button(
            self,
            text="Đóng",
            command=self.destroy,  # Sẽ gọi hàm destroy đã override
            style="TButton"
        )
        self.close_button.pack(pady=15)

    # <<< THÊM VÀO: Ghi đè (override) hàm destroy để đóng lớp phủ >>>
    def destroy(self):
        """Đóng cả cửa sổ dialog và lớp phủ mờ."""
        # Kiểm tra nếu lớp phủ tồn tại thì hủy nó trước
        if hasattr(self, 'overlay') and self.overlay.winfo_exists():
            self.overlay.destroy()

        # Gọi hàm destroy gốc của Toplevel
        super().destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")


    # Tạo một nút để mở dialog
    def open_dialog():
        WasteInfoDialog(root, waste_types=["Nhựa", "Giấy"], waste_info="Nhựa: 2 vật thể\nGiấy: 1 vật thể")


    btn = ttk.Button(root, text="Mở Dialog Test", command=open_dialog)
    btn.pack(pady=50)

    root.mainloop()