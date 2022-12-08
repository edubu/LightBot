"""
    Driver file for the LightBot in EECS467 W22: Autonomous Robotics
"""

import time
import os
import sys

import threading
import cv2
import math
import mediapipe as mp
import numpy as np

from light_bot import LightBot


# Establishing logger for debugging purposes
import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Determines the angle between three points in x,y plane."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = b - c

    if len(a) == 3:
        cosine_angle = np.dot(-1*ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
    else:
        angle = np.arctan2(ba[0]*bc[1] - ba[1]*bc[0], np.dot(ba, bc))

    return np.degrees(angle)


def calculate_distance(landmark1, landmark2):
    """Calculates euclidian distance between two NormalizedLandmark objects."""
    p1 = np.array((landmark1.x, landmark1.y, landmark1.z))
    p2 = np.array((landmark2.x, landmark2.y, landmark2.z))
    return np.linalg.norm(p1 - p2)


def run_holistic(cam_index):
    """Runs mediapipe holistic with pose and hand landmarks drawn."""
    cap = cv2.VideoCapture(cam_index)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()


def run_pose(cam_index):
    """Runs mediapipe pose with all pose connections drawn."""
    cap = cv2.VideoCapture(cam_index)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # # Extract landmarks
            # try:
            #     landmarks = results.pose_landmarks.landmark
            #     left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            #                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            #     right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            #                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            #     right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
            #                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            #     right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
            #                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            #     right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB].x,
            #                    landmarks[mp_pose.PoseLandmark.RIGHT_THUMB].y]
            #     shoulder_angle = calculate_angle(left_shoulder, right_shoulder, right_elbow)
            #     elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            #     wrist_angle = calculate_angle(right_elbow, right_wrist, right_thumb)
            #
            #     # Put angles on image
            #     cv2.putText(image, "right shoulder angle: " + str(shoulder_angle)[0:4],
            #                 (100, 100),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
            #                 )
            #     cv2.putText(image, "right elbow angle: " + str(elbow_angle)[0:4],
            #                 (100, 200),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
            #                 )
            #     cv2.putText(image, "right wrist angle: " + str(wrist_angle)[0:4],
            #                 (100, 300),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
            #                 )
            # except:
            #     pass

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()


fist_reference_points = [(4, 10), (8, 5), (12, 9), (16, 13), (20, 17)]


def is_fist(landmarks) -> bool:
    """Returns True if the given hand landmarks represent a fist."""
    for id1, id2 in fist_reference_points:
        if calculate_distance(landmarks[id1], landmarks[id2]) > 0.1:
            return False

    return True


frame_shape = [720, 1280]
pose_reference_points = [12, 14, 16]
hand_reference_points = [1, 4, 5, 8, 9, 12, 13, 16, 17, 20]


def get_arm_angles(landmarks_front, landmarks_right, hand):
    """
    :param landmarks_front: landmarks generated by the front camera (oriented facing the user)
    :param landmarks_right: landmarks generated by the side camera (oriented facing the right arm from the side)
    :return: tuple (shoulder_theta, shoulder_phi, elbow_angle, wrist_angle)
    Mediapipe Pose generates an estimated z-distance with an origin at the hip. However, we chose to use a second
    camera angle to generate a more accurate z displacement with reference to the front view. Once this data is combined,
    we can extract two angles representing the shoulders displacement in the xy plane (moving the arm out) and the
    yz plane (rotating the shoulder forwards and backwards). We can also generate an elbow and wrist angle.
    """
    joints = []
    for pose_reference_point in pose_reference_points:
        joints.append([
            landmarks_front[pose_reference_point].x,
            landmarks_front[pose_reference_point].y,
            landmarks_right[pose_reference_point].x,
            landmarks_right[pose_reference_point].y
        ])

    shoulder = joints[0]
    elbow = joints[1]
    wrist = joints[2]

    # shoulder angle in xy plane
    shoulder_xy = calculate_angle(
        [shoulder[0], shoulder[1] - 0.1], shoulder[:2], elbow[:2])
    # shoulder angle in yz plane
    shoulder_yz = -1 * \
        calculate_angle([shoulder[2], shoulder[3] - 0.1],
                        shoulder[2:4], elbow[2:4])
    elbow_angle = calculate_angle(shoulder[0:3], elbow[0:3], wrist[0:3])
    wrist_angle = 0
    if hand:
        wrist_angle = -1 * calculate_angle(elbow[2:4], wrist[2:4], hand)

    return shoulder_xy, shoulder_yz, elbow_angle, wrist_angle


def run_mp(cam_index1, cam_index2, isWindows):
    lightbot = LightBot()

    cap0 = None
    cap1 = None
    if isWindows:
        cap0 = cv2.VideoCapture(cam_index1, cv2.CAP_DSHOW)
        cap1 = cv2.VideoCapture(cam_index2, cv2.CAP_DSHOW)
    else:
        cap0 = cv2.VideoCapture(cam_index1)
        cap1 = cv2.VideoCapture(cam_index2)
    caps = [cap0, cap1]

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    # create body key point detector objects.
    pose0 = mp_pose.Pose(min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)
    pose1 = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        loop_start_time = time.time()
        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # crop to 720x720.
        if frame0.shape[1] != 720:
            frame0 = frame0[:, frame_shape[1] // 2 - frame_shape[0] //
                            2:frame_shape[1] // 2 + frame_shape[0] // 2]
            frame1 = frame1[:, frame_shape[1] // 2 - frame_shape[0] //
                            2:frame_shape[1] // 2 + frame_shape[0] // 2]

        # the BGR image to RGB.
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)

        # reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                # only save key points that are indicated in pose_reference_points
                if i not in pose_reference_points:
                    continue

                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                # add keypoint detection points into figure
                cv2.circle(frame0, (pxl_x, pxl_y), 3, (0, 0, 255), -1)

        if results1.pose_landmarks:
            for i, landmark in enumerate(results1.pose_landmarks.landmark):
                # only save key points that are indicated in pose_reference_points
                if i not in pose_reference_points:
                    continue

                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                # add keypoint detection points into figure
                cv2.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)

        if results1.right_hand_landmarks:
            for i, landmark in enumerate(results1.right_hand_landmarks.landmark):
                # only save key points that are indicated in hand_reference_points
                if i not in hand_reference_points:
                    continue

                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                # add keypoint detection points into figure
                cv2.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)

        if results0.pose_landmarks and results1.pose_landmarks:
            hand = None
            fist = False

            if results1.right_hand_landmarks:
                hand = [results1.right_hand_landmarks.landmark[9].x,
                        results1.right_hand_landmarks.landmark[9].y]
                fist = is_fist(results1.right_hand_landmarks.landmark)

            shoulder_xy, shoulder_yz, elbow_angle, wrist_angle = get_arm_angles(results0.pose_landmarks.landmark,
                                                                                results1.pose_landmarks.landmark,
                                                                                hand)

            # Set lightbot joint angles
            api_start_time = time.time()
            joint_angles = [shoulder_yz, shoulder_xy,
                            elbow_angle, wrist_angle, fist]
            lightbot.set_joint_angles(joint_angles)
            api_end_time = time.time()

            cv2.putText(frame0, "right shoulder angle: " + str(shoulder_xy)[0:4],
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,
                                                        0, 0), 2, cv2.LINE_AA
                        )
            cv2.putText(frame0, "right shoulder angle: " + str(shoulder_yz)[0:4],
                        (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,
                                                        0, 0), 2, cv2.LINE_AA
                        )
            cv2.putText(frame0, "right elbow angle: " + str(elbow_angle)[0:4],
                        (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,
                                                        0, 0), 2, cv2.LINE_AA
                        )
            cv2.putText(frame0, "right wrist angle: " + str(wrist_angle)[0:4],
                        (100, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,
                                                        0, 0), 2, cv2.LINE_AA
                        )
            cv2.putText(frame0, "fist: " + str(fist),
                        (100, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,
                                                        0, 0), 2, cv2.LINE_AA
                        )

        # uncomment to draw all landmarks
        # mp_drawing.draw_landmarks(
        #     frame0,
        #     results0.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        #     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        #
        # mp_drawing.draw_landmarks(
        #     frame1,
        #     results1.pose_landmarks,
        #     mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles
        #     .get_default_pose_landmarks_style())
        #
        # mp_drawing.draw_landmarks(
        #     frame1,
        #     results1.right_hand_landmarks,
        #     mp_holistic.HAND_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles
        #     .get_default_pose_landmarks_style())

        cv2.imshow('cam1', frame1)
        cv2.imshow('cam0', frame0)

        loop_end_time = time.time()

        print(
            f'Total frame time: {loop_end_time - loop_start_time}: Robot api comm time: {api_end_time - api_start_time}\n')
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()


if __name__ == "__main__":
    run_mp(0, 2, isWindows=False)
