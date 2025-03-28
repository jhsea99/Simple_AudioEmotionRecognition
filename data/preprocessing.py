import face_alignment
import cv2
import numpy as np
import os
import ffmpeg

def crop_align_face_and_extract_audio(video_path, output_dir, desired_size=(224, 224), padding=0.2):
    """
    Crops and aligns faces from a video, extracts the audio using ffmpeg-python, and saves both.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the cropped and aligned face images and audio.
        desired_size (tuple): The desired size (height, width) of the aligned face images.
        padding (float): Padding around the cropped face (as a fraction of the face size).
    """
    os.makedirs(output_dir, exist_ok=True)
    converted_video_path = "../dataset/example_output/1-30-1280x720" # To store the path of the converted video if needed

    # Check and convert video FPS if necessary
    video_capture_check = cv2.VideoCapture(video_path)
    original_fps = video_capture_check.get(cv2.CAP_PROP_FPS)
    video_capture_check.release()

    if original_fps != 25.0 and original_fps > 0:  # Check if FPS is valid and not already 25
        try:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            converted_video_path = os.path.join(output_dir, f"{base_name}_25fps{os.path.splitext(video_path)[1]}")
            ffmpeg.input(video_path).output(converted_video_path, r=25).run(overwrite_output=True, quiet=True)
            print(f"Video FPS converted from {original_fps:.2f} to 25.0 and saved to: {converted_video_path}")
            video_path = converted_video_path  # Update video path to the converted video
        except ffmpeg.Error as e:
            print(f"Error converting video FPS: {e.stderr.decode('utf8')}")
    # Extract audio using ffmpeg-python
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_output_path = os.path.join(output_dir, f"{base_name}.wav")
        ffmpeg.input(video_path).output(audio_output_path).run(overwrite_output=True)
        print(f"Audio extracted and saved to: {audio_output_path}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio using ffmpeg: {e.stderr.decode('utf8')}")

    fa = face_alignment.FaceAlignment(landmarks_type= face_alignment.LandmarksType.TWO_D, face_detector='sfd', flip_input=False)
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_number = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        frame_number += 1
        try:
            # Detect facial landmarks
            preds = fa.get_landmarks(frame)

            if preds is not None:
                # Assuming only one face is present or we want to process the first detected face
                landmarks = preds[0]

                # Define the eyes center as the reference point for alignment
                left_eye_center = np.mean(landmarks[36:42], axis=0)
                right_eye_center = np.mean(landmarks[42:48], axis=0)

                # Calculate the angle between the eyes
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                angle = np.degrees(np.arctan2(dy, dx))

                # Calculate the center of the face bounding box
                face_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
                face_center_y = (left_eye_center[1] + right_eye_center[1]) // 2

                # Perform affine transformation for alignment
                M = cv2.getRotationMatrix2D((face_center_x, face_center_y), angle, 1.0)
                rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

                # Get the bounding box of the rotated face
                rotated_preds = fa.get_landmarks(rotated_frame)
                if rotated_preds is not None:
                    rotated_landmarks = rotated_preds[0]
                    min_x = int(np.min(rotated_landmarks[:, 0]))
                    max_x = int(np.max(rotated_landmarks[:, 0]))
                    min_y = int(np.min(rotated_landmarks[:, 1]))
                    max_y = int(np.max(rotated_landmarks[:, 1]))

                    # Calculate padding
                    face_width = max_x - min_x
                    face_height = max_y - min_y
                    padding_x = int(face_width * padding)
                    padding_y = int(face_height * padding)

                    # Crop the face with padding
                    crop_min_x = max(0, min_x - padding_x)
                    crop_max_x = min(rotated_frame.shape[1], max_x + padding_x)
                    crop_min_y = max(0, min_y - padding_y)
                    crop_max_y = min(rotated_frame.shape[0], max_y + padding_y)

                    cropped_face = rotated_frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

                    if cropped_face.size > 0:
                        # Resize the cropped face to the desired size
                        resized_face = cv2.resize(cropped_face, desired_size, interpolation=cv2.INTER_LINEAR)

                        # Save the aligned and cropped face
                        output_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
                        cv2.imwrite(output_filename, resized_face)
                        print(f"Processed frame {frame_number}, saved to {output_filename}")
                    else:
                        print(f"Warning: Cropped face is empty for frame {frame_number}")
                else:
                    print(f"Warning: Could not detect face in rotated frame {frame_number}")

            else:
                print(f"Warning: No face detected in frame {frame_number}")

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")

    video_capture.release()
    print("Face cropping and alignment complete.")

if __name__ == "__main__":
    video_path = "../dataset/example/1-30-1280x720.mp4"  # Replace with the actual path to your video
    output_directory = "../dataset/example_output/1-30-1280x720"
    desired_image_size = (224, 224)
    padding_around_face = 0.2

    crop_align_face_and_extract_audio(video_path, output_directory, desired_image_size, padding_around_face)