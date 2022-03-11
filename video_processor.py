import os
import numpy as np
import cv2
from glob import glob
from pytube import YouTube

import re

video_dir = os.path.join('data', 'videos')
frame_dir = os.path.join('data', 'video_frames')

# Download a link from youtube and store the video as an .mp4
# link:         A string in the form of a url
# output path:  Stores the video in the path /data/videos/
def download_video(link):
    # Process the url through pytube
    yt = YouTube(link) 
    title = yt.title

    # convert the title into a valid string (no punctuation)
    valid_chars = "[a-zA-Z0-9\ ]"
    title_formatted = ""
    for char in title.lower().strip():
        if re.match(valid_chars, char):
            title_formatted = title_formatted + char

    # Rename the video based on the first 5 words of the YouTube video's title
    name_arr = title_formatted.split()
    final_name = "_".join(name_arr[0:min(5, len(name_arr))]) + ".mp4"

    # Filter the parameters of the outputed video and download with new name
    try:
        yt.streams.filter(adaptive = True).first().download(output_path = video_dir, filename = final_name)
    except:
        print("Some Error!")
    print('Task Completed!')

# Creates a directory
def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

# Split video into individual frames as .jpgs
# video_path:    The path to find where the video is stored
# gap:           The gap between frame collection. A gap = 1 means all frames are saved,
#                a gap = 24 means every 24 frames are saved.

def save_frame(video_path, gap):
    name = "vf_" + os.path.basename(video_path)[:-4]
    save_path = os.path.join(frame_dir, name)
    create_dir(save_path)
    
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            print("finished extracting frames for", name)
            break

        if idx % gap == 0:
            frame_path = os.path.join(save_path, str(idx) + '.jpg')
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print("error extracting frames")

        idx += 1

if __name__ == "__main__":

    # download_video("https://www.youtube.com/watch?v=jjDYW46PjA8")
    # download_video("https://www.youtube.com/watch?v=V32zWkB1MiY")
    # download_video("https://www.youtube.com/watch?v=zV1qLYukTH8")

    video_paths = [os.path.join(video_dir, filename) for filename in os.listdir(video_dir)]

    for path in video_paths:
        save_frame(path, gap = 40)

    