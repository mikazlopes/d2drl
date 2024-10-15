import numpy as np
import subprocess

# Create a small video data sample
video_data = np.random.randint(0, 255, (30, 64, 64, 3), dtype=np.uint8)

# Simplified command to convert raw video to GIF
cmd = [
    'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-r', '20', '-s', '64x64',
    '-pix_fmt', 'rgb24', '-i', '-', '-vf', 'fps=10,scale=64:64:flags=lanczos', '-c:v', 'gif', '-f', 'gif', 'output.gif'
]

# Start ffmpeg subprocess
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Write video data to stdin
for frame in video_data:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()

# Read stderr to get any errors
stderr = proc.stderr.read()
proc.stderr.close()

# Check for successful execution
if proc.wait() == 0:
    print("ffmpeg processed the video successfully.")
else:
    print("ffmpeg failed with errors:")
    print(stderr.decode())
