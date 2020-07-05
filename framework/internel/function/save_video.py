import cv2

def save_video(frames, path,fps,width,height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()