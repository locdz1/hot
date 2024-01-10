from flask import Flask, request, jsonify
import cv2
import pytesseract
from PIL import Image, ImageOps
import base64
import numpy as np
import re

app = Flask(__name__)

# Đường dẫn tới thư mục chứa tesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Đọc danh sách key từ tệp api_keys.txt
with open('key.txt') as f:
    api_keys = set(line.strip() for line in f)

def is_valid_key(key):
    return key in api_keys

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Nhận dữ liệu ảnh base64 và key từ request
        data = request.json
        base64_image = data.get('base64_image', '')
        api_key = data.get('api_key', '')

        # Kiểm tra tính hợp lệ của key
        if not is_valid_key(api_key):
            return jsonify({"error": "Invalid API key"})

        # Tách phần dữ liệu base64 từ chuỗi
        image_data = base64_image.split(',')[1]

        # Giải mã base64 thành mảng bytes
        decoded_data = base64.b64decode(image_data)

        # Chuyển đổi mảng bytes thành mảng numpy
        numpy_image = np.frombuffer(decoded_data, dtype=np.uint8)

        # Đọc ảnh từ mảng numpy
        image = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)

        # Chuyển đổi ảnh từ OpenCV sang Pillow
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Đảo màu ảnh (white text on black background)
        inverted_image = ImageOps.invert(pil_image)

        # Chuyển đổi ảnh từ Pillow sang mảng numpy
        numpy_image = np.array(inverted_image)

        # Áp dụng Gaussian Blur để loại bỏ nhiễu
        blurred_image = cv2.GaussianBlur(numpy_image, (5, 5), 0)

        # Chuyển đổi ảnh sang ảnh đen trắng (ảnh nhị phân)
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Lưu ảnh đã xử lý (để kiểm tra)
        cv2.imwrite('/var/www/html/binary_image.png', binary_image)

        # Mở ảnh đã xử lý bằng thư viện Pillow
        processed_image = Image.open('/var/www/html/binary_image.png')

        # Trích xuất văn bản từ ảnh đã xử lý
        text_with_symbols = pytesseract.image_to_string(processed_image, config='--psm 6')

        # Loại bỏ tất cả các kí tự không phải là chữ cái và số
        cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text_with_symbols)

        # Trả về kết quả
        return jsonify({"result": cleaned_text})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
