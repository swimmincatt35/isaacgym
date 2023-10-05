import os
import sys
import cv2
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def build_video(dir_path, frame_rate, stitch=False):
    # Get all PNG files in the directory
    all_file_names = [f for f in os.listdir(dir_path) if f.endswith('.png')]

    envs_to_files = defaultdict(list)
    print(envs_to_files)
    for file_name in all_file_names:

        if "cam" in file_name:
            env_identifier = file_name[:8]
        elif "chains" in file_name:
            env_identifier = file_name.split("-")[0] + "-" + file_name.split("-")[1]
        envs_to_files[env_identifier].append(file_name)
    
    video_length = len(envs_to_files[env_identifier])

    if stitch==True:
        files_to_imges = {}

    for env_identifier in envs_to_files:

        file_names = envs_to_files[env_identifier]

        # Sort the file names by their numerical order
        file_names.sort(key=lambda x: str(x.split('.')[0]))

        # Get the first image to determine the video dimensions
        img = cv2.imread(os.path.join(dir_path, file_names[0]))
        height, width, depth = img.shape

        if stitch==False:
            # Initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(dir_path, f'-output-{env_identifier}.mp4')
            video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
            # Write each frame to the video
            for file_name in tqdm(file_names):
                img = cv2.imread(os.path.join(dir_path, file_name))
                video_writer.write(img)
            print(f"...wrote video for {env_identifier}, to {video_path}")

            # Release the video writer
            video_writer.release()
        
        else:
            # [CH] store frames and stitch later
            files_to_imges[env_identifier] = np.empty((len(file_names), height, width, depth))
            for i in tqdm(range(len(file_names))):
                img = cv2.imread(os.path.join(dir_path, file_names[i]))
                files_to_imges[env_identifier][i] = img

    if stitch==True:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(dir_path, f'-output-stitched.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width*len(files_to_imges), height))

        for i in tqdm(range(video_length)):
            stiched_img = np.empty((height, width * len(files_to_imges), depth)).astype(np.uint8)
            j = 0
            for env_identifier in files_to_imges:
                stiched_img[:,j*width:(j+1)*width,:] = files_to_imges[env_identifier][i]
                j+=1
            video_writer.write(stiched_img)
        
        print(f"...wrote stiched video, to {video_path}")

        # Release the video writer
        video_writer.release()



if __name__ == '__main__':
    # Get the directory path and frame rate from command line arguments
    dir_path = sys.argv[1]
    frame_rate = int(sys.argv[2])
    stitch = sys.argv[3].lower() == 'true' 

    # Build the video
    build_video(dir_path, frame_rate, stitch)