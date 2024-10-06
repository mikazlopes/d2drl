from flask import Flask, request, jsonify, Response
import asyncio
import pyautogui
import pygetwindow as gw
import win32gui
import win32con
import time
import numpy as np
import io
from PIL import ImageGrab, Image, ImageOps

app = Flask(__name__)

pyautogui.FAILSAFE = False

async def focus_on_diablo_window():
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
async def screenshot():
    if not await focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400
    
    diablo_window = gw.getWindowsWithTitle("Diablo II")
    window = diablo_window[0]
    x, y, width, height = window.left, window.top, window.width, window.height
    screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
    screenshot = screenshot.resize((400, 300))
    img_byte_arr = io.BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return Response(img_byte_arr, mimetype='image/png')
    

@app.route('/screenshotsmall', methods=['GET'])
async def screenshotsmall():
    if not await focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400

    diablo_window = gw.getWindowsWithTitle("Diablo II")
    if not diablo_window:
        return jsonify(error="Diablo II window not found"), 400
    window = diablo_window[0]
    x, y, width, height = window.left, window.top, window.width, window.height

    # Capture screenshot
    screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))

    # Resize image while maintaining aspect ratio
    target_size = 128
    aspect_ratio = width / height
    if aspect_ratio > 1:
        # Wide image
        resized_height = target_size
        resized_width = int(target_size * aspect_ratio)
    else:
        # Tall image
        resized_width = target_size
        resized_height = int(target_size / aspect_ratio)
    resized_screenshot = screenshot.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


    # Calculate padding
    delta_width = target_size - resized_width
    delta_height = target_size - resized_height
    padding = (max(0, delta_width // 2), max(0, delta_height // 2),
               max(0, delta_width - (delta_width // 2)), max(0, delta_height - (delta_height // 2)))

    # Apply padding
    padded_screenshot = ImageOps.expand(resized_screenshot, padding, fill=(0, 0, 0))

    # Verify size after padding
    if padded_screenshot.size != (target_size, target_size):
        return jsonify(error="Error in resizing and padding image"), 500

    # Convert to byte array for response
    img_byte_arr = io.BytesIO()
    padded_screenshot.save(img_byte_arr, format='PNG')
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

@app.route('/keypress', methods=['POST'])
async def keypress():
    if not await focus_on_diablo_window():
        return jsonify(error="Diablo II window not found"), 400

    data = request.get_json()
    key = data['key']
    pyautogui.keyDown(key)
    await asyncio.sleep(0.05)
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

@app.route('/combined_action', methods=['POST'])
async def combined_action():
    if not await focus_on_diablo_window():
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
        await asyncio.sleep(0.05)
        pyautogui.keyUp(key)

    return jsonify(success=True), 200


