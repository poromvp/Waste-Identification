from utils.App import App
from screens.left_panel import left_panel
from tkinter import *
from tkinter import font

class main_panel():
    def __init__(self):
        window = Tk()
        self.app = App(window)
        window.title("Đồ án phân tích rác thải")

        fontTitle = font.Font(family="Terminal", size=30, weight=font.BOLD)
        lbTitle = Label(window, text="HỆ THỐNG PHÂN LOẠI RÁC THẢI", font=fontTitle)
        lbTitle.grid(row=0, column=0, columnspan=2)
        
        self.leftpanel = left_panel(window)
        self.leftpanel.grid(row=1, column=0, sticky="nsew")
        
        window.rowconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        window.columnconfigure(0, weight=1)
        window.mainloop()