import subprocess
import os

def convert_avi_to_mp4_h264(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.mp4"

    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'slow',     # good balance between speed and quality
        '-crf', '23',          # lower = better quality (18–28 recommended)
        '-pix_fmt', 'yuv420p', # most compatible pixel format
        '-c:a', 'aac',         # standard audio codec
        '-b:a', '128k',        # audio bitrate
        output_path
    ]

    print(f"Converting to standard MP4 (H.264): {output_path}")
    subprocess.run(command)
    print("Conversion complete.")

# Example usage
if __name__ == '__main__':
    convert_avi_to_mp4_h264("ros2/ros2_video_output.avi")
