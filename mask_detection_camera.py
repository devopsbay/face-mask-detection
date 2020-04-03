# -*- coding:utf-8 -*-
import cv2
import time
import argparse

from inference import inference


def detect_masks_video(conf_thresh, output_video_name):
    cap = cv2.VideoCapture(0)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(260, 260),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            # Double conversion required due to color issues
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
            writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection From Camera")
    parser.add_argument('--video-output-path', type=str, default='test_video1.mp4', help='output path for video')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection Threshold')
    args = parser.parse_args()

    detect_masks_video(output_video_name=args.video_output_path, conf_thresh=args.threshold)
