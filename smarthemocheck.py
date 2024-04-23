import os
import cv2
import numpy as np
import time
from scipy.signal import find_peaks

def knee_pt(data):
    # Function to find the knee point of a curve
        diff = np.diff(data)
        d_diff = np.diff(diff)
        return np.argmax(d_diff) + 2

def process_video(video_file, output_folder):
    # Function to extract frames from the video
    def extract_video_frames(video_file, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{output_folder}/frame{count}.jpg", image)
            success, image = vidcap.read()
            print(f"Frame {count} extracted")
            count += 1

    # Function to process start time
    def process_start_time(fname, fps, y, x, r1, r2):
        output_file = f'start_time_{fname}.txt'
        met1 = []
        n = int(fps * 10)

        for start in np.arange(0, n + 1, fps / 10):
            img1 = cv2.imread(f'{fname}/frame{int(start)}.jpg')
            sz = img1.shape[:2]
            if sz[0] == 1080:
                img1 = np.rot90(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 3)
            img1, _ = circle_crops(img1, y, x, r1, 1)
            img1 = img1[y - r2:y + r2, x - r2:x + r2]

            img2 = cv2.imread(f'{fname}/frame{int(start + fps / 10)}.jpg')
            sz = img2.shape[:2]
            if sz[0] == 1080:
                img2 = np.rot90(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 3)
            img2, _ = circle_crops(img2, y, x, r1, 1)
            img2 = img2[y - r2:y + r2, x - r2:x + r2]

            diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
            met_val = np.sum(diff)
            met1.append(met_val)
            print(f'{(start / n) * 100}% complete')

        #print("met1:", met1)
        np.savetxt(output_file, met1)
        return met1

    # Function to process stop time
    def process_stop_time(fname, fps, y, x, r1, r2):
        output_file = f'stop_time_{fname}.txt'
        met2 = []
        n = len(os.listdir(fname)) - 3
        n = (n // fps) * fps - (fps / 10)
        n = int(fps * 10)

        for start in np.arange(0, n + 1, fps / 10):
            img1 = cv2.imread(f'{fname}/frame{int(start)}.jpg')
            sz = img1.shape[:2]
            if sz[0] == 1080:
                img1 = np.rot90(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 3)
            img1, _ = circle_crop(img1, y, x, r1, 0)
            img1 = img1[y - r2:y + r2, x - r2:x + r2]
            img2 = cv2.imread(f'{fname}/frame{int(start + fps / 10)}.jpg')
            sz = img2.shape[:2]
            if sz[0] == 1080:
                img2 = np.rot90(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 3)
            img2, _ = circle_crop(img2, y, x, r1, 0)
            img2 = img2[y - r2:y + r2, x - r2:x + r2]

            diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
            met_val = np.sum(diff)
            met2.append(met_val)
            print(f'{(start / n) * 100}% complete')

        #print("met2:", met2)
        np.savetxt(output_file, met2)
        return met2

    # Helper function for circle crops
    def circlecropbw(img, cx, cy, cr, invert):
        img2 = np.uint8(np.dstack([img, img, img]))
        out, _ = circle_crops(img2, cx, cy, cr, invert)
        out = out[:, :, 1]
        return out
    
    def circle_crops(img, cx, cy, cr, invert):
        rows, columns, _ = img.shape
        rgbImage2 = np.zeros((rows, columns, 3), dtype=np.uint8)

        ci = [cx, cy, cr]
        xx, yy = np.meshgrid(np.arange(1, columns + 1) - ci[0], np.arange(1, rows + 1) - ci[1])
        mask = (xx * 2 + yy * 2) < ci[2] ** 2
        if invert == 1:
            mask = ~mask

        redChannel1 = img[:, :, 0]
        greenChannel1 = img[:, :, 1]
        blueChannel1 = img[:, :, 2]
        redChannel2 = rgbImage2[:, :, 0]
        greenChannel2 = rgbImage2[:, :, 1]
        blueChannel2 = rgbImage2[:, :, 2]

        redChannel2[mask] = redChannel1[mask]
        greenChannel2[mask] = greenChannel1[mask]
        blueChannel2[mask] = blueChannel1[mask]

        out = np.stack([redChannel2, greenChannel2, blueChannel2], axis=2)
        return out, mask
    
    
    def circlecropbw(img, cx, cy, cr, invert):
        img2 = np.uint8(np.dstack([img, img, img]))
        out, _ = circle_crops(img2, cx, cy, cr, invert)
        out = out[:, :, 1]
        return out
    
    def circle_crop(img, cx, cy, cr, invert):
        rows, columns, _ = img.shape
        rgbImage2 = np.zeros((rows, columns, 3), dtype=np.uint8)

        ci = [cx, cy, cr]
        xx, yy = np.meshgrid(np.arange(1, columns + 1) - ci[0], np.arange(1, rows + 1) - ci[1])
        mask = (xx * 2 + yy * 2) < ci[2] ** 2
        if invert == 1:
            mask = ~mask

        redChannel1 = img[:, :, 0]
        greenChannel1 = img[:, :, 1]
        blueChannel1 = img[:, :, 2]
        redChannel2 = rgbImage2[:, :, 0]
        greenChannel2 = rgbImage2[:, :, 1]
        blueChannel2 = rgbImage2[:, :, 2]

        redChannel2[mask] = redChannel1[mask]
        greenChannel2[mask] = greenChannel1[mask]
        blueChannel2[mask] = blueChannel1[mask]

        out = np.stack([redChannel2, greenChannel2, blueChannel2], axis=2)
        return out, mask
    
    
    

    def compute_pt():
        fname = 'output_folder'
        fps = 60
        y, x = 920, 500
        r1, r2 = 200, 350

# Read particle motion curve and note length of video
        end_video = np.loadtxt('stop_time_' + fname + '.txt')
        end_video = np.convolve(end_video, np.ones(10), mode='valid') / 10

        sz = end_video.shape
        t = np.linspace(0, (len(end_video) * (fps / 10)) / fps, sz[0])

# Calculate start time
# Read start_time file
        begin_video = np.loadtxt('start_time_' + fname + '.txt')

# Find knee point of pipette motion curve
        kp = knee_pt(np.convolve(begin_video, np.ones(10), mode='valid'))

# Crop from start to knee point
        begin_video = begin_video[:kp]
        xstart = np.linspace(0, (len(begin_video) * (fps / 10)) / fps, len(begin_video))

# Find most prominent peak in cropped range
# Then identify start time
#start_time_metrics = process_start_time(fname, fps, y, x, r1, r2)
        pks, locs = find_peaks(begin_video)
        if len(locs) > 0:
            startp = np.argmax(pks)
            startloc = locs[startp]
            begin_time = t[startloc]
        else:
            begin_time = 0

# Calculate end time
# Offset the start point by 10 seconds from when measurement starts
#stop_time_metrics = process_stop_time(fname, fps, y, x, r1, r2)
        offset = startloc + (10 * 10) if len(locs) > 0 else 0
        end_video = end_video[offset:]

# Normalize motion curve between [0,1]
# Trim off end of motion curve
#np.savetxt(f'start_time_{fname}.txt', start_time_metrics)
#np.savetxt(f'stop_time_{fname}.txt', stop_time_metrics)
        end_video = (end_video - np.min(end_video)) / (np.max(end_video) - np.min(end_video))
        f = np.where(end_video < 0.01)[0]
        end_video = end_video[:f[0]]

        kp = knee_pt(end_video) + offset
        end_time = t[kp]

# Calculate PT and INR
        pt = (end_time - begin_time) * 10  # Convert time to seconds
        pt_normal = 12  # Assuming this is the normal PT value in seconds
        isi = 1.31
        alpha = -0.31
        inr = (pt / pt_normal) ** (isi - alpha)

# Define normal ranges
        normal_pt_range = (11, 13.5)
        normal_inr_range = (0.8, 1.1)

# Check if PT and INR are within normal ranges
        if normal_pt_range[0] <= pt <= normal_pt_range[1]:
            pt_status = "Normal"
        else :
            pt_status = "Abnormal Treatment required"

        if normal_inr_range[0] <= inr <= normal_inr_range[1]:
            inr_status = "Normal"
        else:
            inr_status = "Abnormal Treatment required"

        print('PT: {:.1f} seconds ({})'.format(pt, pt_status))
        print('INR: {:.3f} ({})'.format(inr, inr_status))
    
    # Start of process_video function
    start_time = time.time()
    extract_video_frames(video_file, output_folder)
    process_start_time(output_folder, 60, 920, 500, 200, 350)
    process_stop_time(output_folder, 60, 920, 500, 200, 350)
    compute_pt()
    print("Total processing time:", time.time() - start_time)


# Example usage:
process_video('video.mp4', 'output_folder')