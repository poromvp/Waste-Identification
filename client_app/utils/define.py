import os

PATH_DIRECTORY = os.path.dirname(os.path.dirname(__file__))  # Lấy đường dẫn thư mục hiện tại
PATH_IMAGES = os.path.join(PATH_DIRECTORY, "assets", "image")  # Tạo đường dẫn đến thư mục image
PATH_ICON = os.path.join(PATH_DIRECTORY, "assets", "icon") # Tạo đường dẫn đến thư mục icon
COLOR_BACKGROUND = "#7DF4AB"  # Màu nền của ứng dụng (Sky Blue)