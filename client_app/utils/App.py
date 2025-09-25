from utils.define import *
import os
class App():
    def __init__(self, window):
        # Thiết lập kíck thước (width x height + x_offset + y_offset)
        #root.geometry("800x400+300+300")

        # Lấy kích thước màn hình
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # Kích thước cửa sổ
        window_width = 800
        window_height = 400

        # Tính toán vị trí để căn giữa cửa sổ
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Thiết lập kích thước và vị trí cửa sổ
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        #Title
        window.title("CHANGE COLOR")

        #Thay đổi icon của cửa sổ
        window.iconbitmap(os.path.join(PATH_ICON, "icon_xe_rac.ico"))

        #Thay đổi màu nền
        #root['bg'] = "#87CEEB"
        #root['background'] = "#87CEEB"
        window.configure(bg=COLOR_BACKGROUND)

        #Không cho phép thay đổi kích thước cửa sổ
        #window.resizable(False, False) # (width, height)
        window.minsize(600, 300)
        window.maxsize(1200, 800)