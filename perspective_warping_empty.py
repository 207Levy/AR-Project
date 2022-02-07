# ======= imports
import cv2
import numpy as np
from os import system

# ======= constants
is_downsample = True

# === template image keypoint and descriptors
target = cv2.imread("target.jpg")
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)


forest_img = cv2.imread('forest.jpg')
forest_img_rgb = cv2.cvtColor(forest_img, cv2.COLOR_BGR2RGB)
forest_img_resize = cv2.resize(forest_img_rgb, (target.shape[1], target.shape[0]))



# find the keypoints and descriptors with chosen feature_extractor
sift = cv2.SIFT_create()
kp_target, desc_target = sift.detectAndCompute(target_gray, None)

# ===== video input
video = cv2.VideoCapture("target.MOV")
fps = int(video.get(cv2.CAP_PROP_FPS))
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:", fps)
print("lenth:", length)

# ===== video write
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('wrap.avi', fourcc, fps, (width, height))

# ========== run on all frames
frame_num = -1
ls = 0
while video.isOpened():
    system('cls')
    success, frame = video.read()
    if not success:
        print("can't read frame")
        break

    frame_num += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ====== find keypoints matches of frame and template
    kp_frame, desc_frame = sift.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_target, desc_frame, k=2)

    # Ratio test
    good_match_list = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.5:
            good_match_list.append(m)
    good_match_arr = np.asarray(good_match_list)[:, 0]



    if good_match_arr.size < 20:
        print("skip frame " + str(frame_num))
        continue

    if is_downsample:
        if frame_num % 10 == 0:
            continue

    # ======== find homography
    good_kp_target = np.array([kp_target[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])
    H_matrix, mask = cv2.findHomography(good_kp_target, good_kp_frame, cv2.RANSAC, 5.0)
    if not isinstance(H_matrix, np.ndarray):
        print("skip frame")
        continue

    # ========= do warping of another image on template image
    mask_warped = cv2.warpPerspective(np.ones(target.shape, dtype=np.uint8), H_matrix, (frame_rgb.shape[1], frame_rgb.shape[0]))

    mask_warped_bin = mask_warped > 0
    im_warped = cv2.warpPerspective(forest_img_resize, H_matrix, (frame_rgb.shape[1], frame_rgb.shape[0]))
    frame_rgb[mask_warped_bin] = im_warped[mask_warped_bin]
    # =========== plot and save frame

    out.write(frame_rgb)
    status = int((frame_num/length) * 100)
    if ls != status:
        print(str(status) + "% ...")
    ls = status
# ======== end all
video.release()
cv2.destroyAllWindows()
print("====== finished ======")