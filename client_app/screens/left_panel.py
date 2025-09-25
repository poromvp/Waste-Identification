from utils.define import *
import tkinter as tk

class left_panel():
    def __init__(self, window):
        # Tạo canvas kích thước 400x300, nền trắng
        canvas = tk.Canvas(window, width=400, height=300, bg="white")

        # Vẽ một số hình
        canvas.create_line(10, 10, 200, 50, fill="blue", width=3)     # Vẽ đường thẳng
        canvas.create_rectangle(50, 100, 150, 200, fill="red")        # Vẽ hình chữ nhật
        canvas.create_oval(200, 100, 300, 200, fill="green")          # Vẽ hình tròn/ellipse
        canvas.create_text(200, 250, text="Hello Canvas", font=("Arial", 16), fill="purple")