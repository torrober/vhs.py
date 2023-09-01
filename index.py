import cv2
import easygui
import numpy as np
import base64
from vhs import VHS
import eel
from pydub import AudioSegment
from pydub.playback import play
import  tkinter as tk
from tkinter import filedialog
params = []
@eel.expose
def fileChooser():
    root = tk.Tk()
    root.withdraw()

    # Create a top-level window for the file dialog
    top_level = tk.Toplevel(root)

    # Make the file dialog window a transient window
    top_level.transient(root)

    file_path = filedialog.askopenfilename(parent=top_level)
    recordToTape(file_path)



def recordToTape(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    loadTape = VHS(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])  # Define your VHS effect here
    # Load the noise video and set it to loop
    noise = cv2.VideoCapture("noise-preprocess.avi")
    noise_fps = int(noise.get(cv2.CAP_PROP_FPS))
    noise_frames = int(noise.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    # Loop through video frames
    noise_frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply VHS effect to the frame
        frame = cv2.resize(frame, (640, 480))
        frame = loadTape.applyVHSEffect(frame)

        # Write the frame to the output video
        out.write(frame)

        # Preview the output frame
        cv2.imshow('VHS Effect', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit the preview
            break

    # Release video capture and writer
    cap.release()
    noise.release()
    out.release()
    cv2.destroyAllWindows()

def combine_audio(video_path, audio_path):
    import moviepy.editor as mp

    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)

    final_video = video.set_audio(audio)
    final_video.write_videofile('final_output_video.mp4', codec='libx264')

@eel.expose
def updateFrame(image, lumaCompressionRate, lumaNoiseSigma, lumaNoiseMean, chromaCompressionRate, verticalBlur, horizontalBlur, chromaNoiseIntensity, borderSize, generations):
    global params
    params = [lumaCompressionRate, lumaNoiseSigma, lumaNoiseMean, chromaCompressionRate, chromaNoiseIntensity, verticalBlur,horizontalBlur, borderSize]
    loadTape = VHS(lumaCompressionRate, lumaNoiseSigma, lumaNoiseMean, chromaCompressionRate, chromaNoiseIntensity, verticalBlur,horizontalBlur, borderSize)
    loadTape.generation = generations
    decoded_image = base64.b64decode(image)
    image_array = np.frombuffer(decoded_image, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    frame = loadTape.applyVHSEffect(img)
    _, buffer = cv2.imencode('.png', frame)
    base64_image = base64.b64encode(buffer).decode()
    eel.setImage(base64_image)
@eel.expose
def on_windows_close():
    quit()
eel.init('ui')
eel.start('index.html')
#recordToTape('test.MP4')