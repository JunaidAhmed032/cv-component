import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib.pyplot as plt
########################Video Capture###################
mp_draw=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
count=0
#level=None
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
def bicepsCurls_Right(count):
    capture = cv2.VideoCapture(0)
    level = None
    with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
        while capture.isOpened():
            ret, frame = capture.read()
            # changeing image color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Making detection
            res = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ########################################
            # extract landmarks
            try:
                landmarks = res.pose_landmarks.landmark
                # Get coordinates
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle

                angle = calculate_angle(shoulder_r, elbow_R, wrist_R)




                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow_R, [780, 240]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 160:
                    level = "down"
                if angle < 45 and level == 'down':
                    level = "up"
                    count += 1
                    print(count)


            except:
                pass

            #  cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
            cv2.putText(image, 'COUNT', (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEVEL', (100, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, level,
                        (95, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(res)
            cv2.imshow('Right side Bicep Curls', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()


def bicepsCurls_left(count):
    capture = cv2.VideoCapture(0)
    level = None
    with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
        while capture.isOpened():
            ret, frame = capture.read()
            # changeing image color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Making detection
            res = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ########################################
            # extract landmarks
            try:
                landmarks = res.pose_landmarks.landmark
                # Get coordinates
                shoulderl = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle

                angle = calculate_angle(shoulderl, elbow_l, wrist_l)




                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow_l, [780, 240]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 160:
                    level = "down"
                if angle < 35 and level == 'down':
                    level = "up"
                    count += 1
                    print(count)


            except:
                pass

            #  cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
            cv2.putText(image, 'COUNT', (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEVEL', (100, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, level,
                        (95, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(res)
            cv2.imshow('Left Side Bicep_curls', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
def Situps(count):
    capture = cv2.VideoCapture('c.mp4')
    level = None
    with mp_pose.Pose(min_tracking_confidence=0.8, min_detection_confidence=0.9) as pose:
        while capture.isOpened():
            ret, frame = capture.read()
            # changeing image color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Making detection
            res = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ########################################
            # extract landmarks
            try:
                landmarks = res.pose_landmarks.landmark
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, hip, knee)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(hip, [780, 240]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 100:
                    level = "down"
                if angle < 35 and level == 'down':
                    level = "up"
                    count += 1
                    print(count)

            except:
                pass

            #  cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
            cv2.putText(image, 'COUNT', (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEVEL', (100, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, level,
                        (95, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(res)
            cv2.imshow('Situps', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
def Squats(count):
    capture = cv2.VideoCapture('squats.mp4')
    level = None
    with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
        while capture.isOpened():
            ret, frame = capture.read()
            # changeing image color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Making detection
            res = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ########################################
            # extract landmarks
            try:
                landmarks = res.pose_landmarks.landmark
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [780, 240]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )


                if angle > 160:
                    level = "up"
                if angle < 70 and level == 'up':
                    level = "down"
                    count += 1
                    print(count)

            except:
                pass

            #  cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
            cv2.putText(image, 'COUNT', (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEVEL', (100, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, level,
                        (95, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(res)
            cv2.imshow('Squats', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
def Pushups(count):
    capture = cv2.VideoCapture('pushup.mp4')
    level = None
    with mp_pose.Pose(min_tracking_confidence=0.8, min_detection_confidence=0.9) as pose:
        while capture.isOpened():
            ret, frame = capture.read()
            # changeing image color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Making detection
            res = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ########################################
            # extract landmarks
            try:
                landmarks = res.pose_landmarks.landmark
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [780, 240]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 160:
                    level = "up"
                if angle < 70 and level == 'up':
                    level = "down"
                    count += 1
                    print(count)

            except:
                pass

            #  cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
            cv2.putText(image, 'COUNT', (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEVEL', (100, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, level,
                        (95, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(res)
            cv2.imshow('PushUPS', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
def Jumingjacks(count):
    level=None
    capture=cv2.VideoCapture('Jumping Jacks.mp4')
    with mp_pose.Pose(min_tracking_confidence=0.8, min_detection_confidence=0.9) as pose:
        while capture.isOpened():
            ret, frame = capture.read()
            # changeing image color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Making detection
            res = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ########################################
            # extract landmarks
            try:
                landmarks = res.pose_landmarks.landmark
                # Get coordinates
                shoulderl = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(wrist_l, shoulderl, hip_l)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(shoulderl, [780, 240]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 165:
                    level = "down"
                if angle < 15 and level == 'down':
                    level = "up"
                    count += 1
                    print(count)

            except:
                pass

            #  cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
            cv2.putText(image, 'COUNT', (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEVEL', (100, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, level,
                        (95, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(res)
            cv2.imshow('Juming Jacks', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
def main():
    print("Welcome to the AR Trainer\n Select the Below Exersice to Start:\n1:Bicep Curls: \n2:Situps: \n3:Pushups: \n4:Squats: \n5:Jumping Jacks: ")
    choice=int(input("Enter Exersice Number"))
    if(choice==1):
        side=input("enter Side")
        if(side=="left"):
            bicepsCurls_left(count)
        else:
            bicepsCurls_Right(count)
    if(choice==2):
        Situps(count)
    if(choice==3):
        Pushups(count)
    if(choice==4):
        Squats(count)
    if(choice==5):
        Jumingjacks(count)
    else:
        print("sorry were not able to perform this Exersice Yet")

if __name__ == "__main__":
    main()

