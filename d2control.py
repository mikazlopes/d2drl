from flask import Flask, request, jsonify, Response
import asyncio
import pyautogui
import pygetwindow as gw
import win32gui
import win32con
import time
import numpy as np
import io
from PIL import ImageGrab, Image

app = Flask(__name__)

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

@app.route('/screenshotreset', methods=['GET'])
async def screenshot400():
    if not focus_on_diablo_window():
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

