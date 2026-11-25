import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


NUM_FRAMES = 32
NUM_FEATURES = 258
MAX_IMAGE_SIZE = 960


def resize_frame(frame):
    height, width = frame.shape[:2]

    largest = max(height, width)

    if largest > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / largest

        frame = cv2.resize(
            frame,
            (
                int(width * scale),
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    return frame


def extract(path, model):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError(f"Invalid video: {path}")

    wanted = sorted(
        set(
            np.linspace(
                0,
                total - 1,
                NUM_FRAMES
            ).astype(int)
        )
    )

    features = []

    pose_count = 0
    left_count = 0
    right_count = 0

    frame_index = 0
    wanted_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if (
            wanted_index < len(wanted)
            and frame_index == wanted[wanted_index]
        ):

            frame = resize_frame(frame)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            result = model.process(rgb)

            values = []

            # Pose landmarks: 33 x 4 = 132

            if result.pose_landmarks:

                pose_count += 1

                for point in result.pose_landmarks.landmark:

                    values.extend([
                        point.x,
                        point.y,
                        point.z,
                        point.visibility
                    ])

            else:

                values.extend([0.0] * 132)

            # Left hand: 21 x 3 = 63

            if result.left_hand_landmarks:

                left_count += 1

                for point in result.left_hand_landmarks.landmark:

                    values.extend([
                        point.x,
                        point.y,
                        point.z
                    ])

            else:

                values.extend([0.0] * 63)

            # Right hand: 21 x 3 = 63

            if result.right_hand_landmarks:

                right_count += 1

                for point in result.right_hand_landmarks.landmark:

                    values.extend([
                        point.x,
                        point.y,
                        point.z
                    ])

            else:

                values.extend([0.0] * 63)

            features.append(values)

            wanted_index += 1

        frame_index += 1

        if wanted_index >= len(wanted):
            break

    cap.release()

    # Pad videos containing fewer than 32 sampled frames.

    array = np.asarray(
        features,
        dtype=np.float32
    )

    if len(array) < NUM_FRAMES:

        padding = np.zeros(
            (
                NUM_FRAMES - len(array),
                NUM_FEATURES
            ),
            dtype=np.float32
        )

        array = np.concatenate(
            [array, padding],
            axis=0
        )

    array = array[:NUM_FRAMES]

    detections = {
        "pose": pose_count,
        "left_hand": left_count,
        "right_hand": right_count,
        "valid_frames": int(
            np.any(array != 0, axis=1).sum()
        )
    }

    return array, detections


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()

    with manifest_path.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    if args.limit > 0:
        rows = rows[:args.limit]

    X = []
    y = []
    signer = []

    failed = []
    empty = []

    holistic = mp.solutions.holistic

    print("\nInitializing MediaPipe Holistic...")

    with holistic.Holistic(

        static_image_mode=True,

        model_complexity=1,

        enable_segmentation=False,

        min_detection_confidence=0.3,

        min_tracking_confidence=0.3

    ) as model:

        for index, row in enumerate(rows, start=1):

            path = Path(row["video_path"])

            if not path.is_absolute():
                path = manifest_path.parent / path

            print(
                f"\n[{index}/{len(rows)}] "
                f"Extracting: {path.name}"
            )

            try:

                landmarks, stats = extract(
                    path,
                    model
                )

                print(
                    f"Pose: {stats['pose']}/32 | "
                    f"Left hand: {stats['left_hand']}/32 | "
                    f"Right hand: {stats['right_hand']}/32"
                )

                print(
                    f"Valid frames: "
                    f"{stats['valid_frames']}/32"
                )

                if stats["valid_frames"] == 0:

                    empty.append(str(path))

                    print("WARNING: No landmarks detected.")

                X.append(landmarks)

                y.append(row["label"])

                signer.append(
                    row.get("signer_id", "unknown")
                )

            except Exception as error:

                failed.append(
                    (str(path), str(error))
                )

                print("ERROR:", error)

    if not X:

        raise RuntimeError(
            "No videos were successfully processed."
        )

    X = np.stack(X)

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(

        output,

        X=X,

        y=np.asarray(y),

        signer=np.asarray(signer)

    )

    print("\n========== EXTRACTION SUMMARY ==========")

    print("Videos processed:", len(X))

    print("Empty videos:", len(empty))

    print("Failed videos:", len(failed))

    print("Feature shape:", X.shape)

    print("Saved:", output.resolve())

    if empty:

        print("\nVideos with zero detections:")

        for video in empty:
            print(video)

    if failed:

        print("\nVideos that failed:")

        for video, error in failed:
            print(video, ":", error)


if __name__ == "__main__":
    main()