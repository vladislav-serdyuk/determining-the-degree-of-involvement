import math
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend/app/'))
from services.video_processing import HeadPoseEstimator

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

CALIBRATION_MATRIX = np.array(
    [
        [1400.0, 0.0, 640.0],
        [0.0, 1400.0, 360.0],
        [0.0, 0.0, 1.0],
    ]
)
DISTORTION_COEFFS = np.zeros((5, 1))

MARKER_LENGTH = 0.05

MARKER_OBJECT_POINTS = np.array(
    [
        [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0.0],
        [-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0.0],
        [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0.0],
        [MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0.0],
    ],
    dtype=np.float32,
)


def rotation_vector_to_euler_angles(rotation_vector: np.ndarray) -> tuple[float, float, float]:
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    pitch = x * 180.0 / math.pi
    yaw = y * 180.0 / math.pi
    roll = z * 180.0 / math.pi

    return pitch, yaw, roll


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть веб-камеру")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    head_pose_estimator = HeadPoseEstimator()

    print("Нажмите 'q' для выхода")

    diffs_pitch = []
    diffs_yaw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break

        image_height, image_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        aruco_results = None
        face_mesh_results = face_mesh.process(rgb_frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs = []
            tvecs = []
            for corner in corners:
                success, rvec, tvec = cv2.solvePnP(
                    MARKER_OBJECT_POINTS,
                    corner,
                    CALIBRATION_MATRIX,
                    DISTORTION_COEFFS,
                )
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)

            pitch, yaw, roll = rotation_vector_to_euler_angles(rvecs[0])
            aruco_results = {'pitch': pitch, 'yaw': yaw, 'roll': roll}

            cv2.drawFrameAxes(frame, CALIBRATION_MATRIX, DISTORTION_COEFFS, rvecs[0], tvecs[0], 0.03)
            cv2.putText(
                frame,
                f"ArUco: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            print(f"ArUco: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}")

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            head_pose = head_pose_estimator.estimate(face_landmarks, image_width, image_height)

            if head_pose:
                roll = head_pose.roll
                pitch = head_pose.pitch
                yaw = head_pose.yaw

                color = (255, 0, 0) if aruco_results is None else (0, 255, 255)
                cv2.putText(
                    frame,
                    f"FaceMesh: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                print(f"FaceMesh: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}")

                if aruco_results:
                    diff_pitch = abs(pitch - aruco_results['pitch'])
                    diff_yaw = abs(yaw - aruco_results['yaw'])
                    diff_roll = abs(roll - aruco_results['roll'])
                    cv2.putText(
                        frame,
                        f"Diff: P:{diff_pitch:.1f} Y:{diff_yaw:.1f} R:{diff_roll:.1f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )
                    print(f"Diff: P:{diff_pitch:.1f} Y:{diff_yaw:.1f} R:{diff_roll:.1f}")
                    diffs_pitch.append(diff_pitch)
                    diffs_yaw.append(diff_yaw)
        print('---')
        cv2.imshow("ArUco vs FaceMesh Head Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f'MAE pitch:', sum(diffs_pitch) / len(diffs_pitch))
    print(f'MAE yaw:', sum(diffs_yaw) / len(diffs_yaw))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
