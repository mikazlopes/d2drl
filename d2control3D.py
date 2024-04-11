from flask import Flask, request, jsonify, Response
import pyautogui
import pygetwindow as gw
import win32gui
import win32con
import io
import time
from PIL import ImageGrab, Image, ImageOps

app = Flask(__name__)

pyautogui.FAILSAFE = False

def focus_on_diablo_window():
    window_name = "Diablo II"
    handle = win32gui.FindWindow(None, window_name)
    if handle == 0:
        return False

    try:
        win32gui.ShowWindow(handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(handle)
        rect = win32gui.GetWindowRect(handle)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        win32gui.MoveWindow(handle, 0, 0, width, height, True)
        return True
    except Exception as e:
        print(f"An error occurred while focusing on the Diablo II window: {e}")
        return False

@app.route('/screenshot', methods=['GET'])
def screenshot():
    if not focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400

    diablo_window = gw.getWindowsWithTitle("Diablo II")
    if not diablo_window:
        return jsonify(error="Diablo II window not found"), 400
    window = diablo_window[0]
    x, y, width, height = window.left, window.top, window.width, window.height
    screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
    screenshot = screenshot.resize((200, 150), Image.Resampling.LANCZOS)
    img_byte_arr = io.BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return Response(img_byte_arr, mimetype='image/png')

@app.route('/screenshotsmall', methods=['GET'])
def screenshotsmall():
    if not focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400

    diablo_window = gw.getWindowsWithTitle("Diablo II")
    if not diablo_window:
        return jsonify(error="Diablo II window not found"), 400
    window = diablo_window[0]
    bbox = window.left, window.top, window.right, window.bottom

    # Capture screenshot
    screenshot = ImageGrab.grab(bbox=bbox)

    # Resize image while maintaining aspect ratio
    target_size = 64
    screenshot.thumbnail((target_size, target_size))

    # Calculate padding to get to 128x128
    width, height = screenshot.size
    padding = (target_size - width) // 2, (target_size - height) // 2
    padding = (padding[0], padding[1], target_size - width - padding[0], target_size - height - padding[1])

    # Apply padding and get the final image
    final_image = ImageOps.expand(screenshot, padding, fill=0)  # Fill with black

    # Convert to byte array for response
    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return Response(img_byte_arr, mimetype='image/png')


@app.route('/screenshotreset', methods=['GET'])
async def screenshotreset():
    if not await focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400
    diablo_window = gw.getWindowsWithTitle("Diablo II") 

    window = diablo_window[0]

    # Get the window's coordinates
    x, y, width, height = window.left, window.top, window.width, window.height
    
    # Capture the screenshot
    screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
    
    # Process the image (e.g., convert to grayscale)
    screenshot = screenshot.resize((400, 300))
    screenshot_gray = screenshot.convert('L')

    # Save to a BytesIO object
    img_byte_arr = io.BytesIO()
    screenshot_gray.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Go to the start of the BytesIO object

    # Send the image as a response
    return Response(img_byte_arr, mimetype='image/png')

@app.route('/combined_action', methods=['POST'])
def combined_action():
    if not focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400

    data = request.get_json()

    # Handle mouse move
    mouse_move_data = data.get('mouse_move_action')
    if mouse_move_data:
        x, y = mouse_move_data['x'], mouse_move_data['y']
        pyautogui.moveTo(x, y)

    # Handle mouse click
    mouse_click_data = data.get('mouse_click_action')
    if mouse_click_data:
        button = mouse_click_data['button']
        pyautogui.click(button=button)

    # Handle key press
    keypress_data = data.get('keypress_action')
    if keypress_data:
        key = keypress_data['key']
        pyautogui.keyDown(key)
        time.sleep(0.05)  # Replace asyncio.sleep with time.sleep
        pyautogui.keyUp(key)

    return jsonify(success=True), 200

@app.route('/keypress', methods=['POST'])
async def keypress():
    if not await focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400

    data = request.get_json()
    key = data['key']
    pyautogui.keyDown(key)
    time.sleep(0.05)  # Replace asyncio.sleep with time.sleep
    pyautogui.keyUp(key)
    return jsonify(success=True), 200

@app.route('/mouse', methods=['POST'])
async def mouse():
    if not await focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400

    data = request.get_json()
    action = data['action']
    window = gw.getWindowsWithTitle("Diablo II")[0]
    
    if action == 'move':
        x, y = data['x'], data['y']
        x = max(window.left, min(window.left + window.width, x))
        y = max(window.top, min(window.top + window.height, y))
        pyautogui.moveTo(x, y)
    elif action == 'click':
        button = data['button']
        pyautogui.click(button=button)
    elif action == 'scroll':
        amount = data['amount']
        pyautogui.scroll(amount)
    return jsonify(success=True), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=True)
