# ======= imports
import cv2
import numpy as np
from os import system
import mesh_renderer

# ======= constants
figsize = (15, 15)
KeyPoint_TRESH = 10
is_downsample = False

# === template image keypoint and descriptors
target = cv2.imread("target.jpg")
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_target, desc_target = sift.detectAndCompute(target_gray, None)

# ==== forest AR ground
forest_img = cv2.imread('forest.jpg')
forest_img_rgb = cv2.cvtColor(forest_img, cv2.COLOR_BGR2RGB)
forest_img_resize = cv2.resize(forest_img_rgb, (target.shape[1], target.shape[0]))

# ===== video input, output and metadata
video = cv2.VideoCapture("target.MOV")
fps = int(video.get(cv2.CAP_PROP_FPS))
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:", fps)
print("lenth:", length)
print("w:", width)
print("h:", height)

# ===== video write
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('final.avi', fourcc, fps, (width, height))

# ===== camera calibration METADATA ===>> from "./calib_chess/clibration data.txt"

RMS = 0.8470934589416081

cam_matrix = np.array( [[1.68769662e+03, 0.00000000e+00, 9.24113165e+02],
                        [0.00000000e+00, 1.62796480e+03, 4.65166299e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs = np.array([1.82293496e-01, -4.70593759e-01, -3.52919002e-04, -1.07468922e-02, 1.17186427e+00])


sqrSize = 5.56
target_real_w = 15.2
target_real_h = 22.5

# adding 3d object on ground with meshRenrer
print("making 3d objects\n")

wraper_first = mesh_renderer.MeshRenderer(cam_matrix, width, height, "Trunk wood_obj/trunk wood.obj", 0)
wraper_second = mesh_renderer.MeshRenderer(cam_matrix, width, height, "OBJ/Campfire.obj", 1)
wraper_third = mesh_renderer.MeshRenderer(cam_matrix, width, height, "DeadTree/DeadTree.obj", 2)

print("done!")

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

    # ====== find keypoints matches of frame and target
    kp_frame, desc_frame = sift.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_target, desc_frame, k=2)

    # Ratio test
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]


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

    # ========= do warping of another image on target image
    #bigger_ground = (target.shape[0]*3, target.shape[1]*3, target.shape[2])
    mask_warped = cv2.warpPerspective(np.ones(target.shape, dtype=np.uint8), H_matrix, (frame_rgb.shape[1], frame_rgb.shape[0]))
    mask_warped_bin = mask_warped > 0
    im_warped = cv2.warpPerspective(forest_img_resize, H_matrix, (frame_rgb.shape[1], frame_rgb.shape[0]))
    frame_rgb[mask_warped_bin] = im_warped[mask_warped_bin]

    # ++++++++ take subset of keypoints that obey homography (both frame and target)
    kp_sub_indx = (np.array(mask).flatten() > 0)
    sub_kp_target = np.array(good_kp_target)[kp_sub_indx, :]
    best_kp_frame = np.array(good_kp_frame.reshape(good_kp_frame.shape[0], 2))[kp_sub_indx, :]

    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera matrix
    # - camera dist_coeffs
    # output
    # camera pose: (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows

    sub_kp_target_3d_cm = np.array([[x[0] / target_gray.shape[1] * target_real_w, x[1] / target_gray.shape[0] * target_real_h, 0] for x in sub_kp_target])
    res, rvec, tvec = cv2.solvePnP(sub_kp_target_3d_cm, best_kp_frame, cam_matrix, dist_coeffs)

    # ++++++ draw object with r_vec and t_vec on top of rgb frame

    drawn_image = wraper_first.draw(frame_rgb, rvec, tvec)
    drawn_image2 = wraper_second.draw(drawn_image, rvec, tvec)
    drawn_image3 = wraper_third.draw(drawn_image2, rvec, tvec)

    # =========== plot and save frame
    final_frame = cv2.cvtColor(drawn_image3, cv2.COLOR_RGB2BGR)
    #cv2.imshow("frame", final_frame)
    out.write(final_frame)

    # ======= output status
    status = int((frame_num/length) * 100)
    if ls != status:
        print(str(status) + "% ...")
    ls = status
# ======== end all
video.release()
cv2.destroyAllWindows()
print("====== finished ======")

