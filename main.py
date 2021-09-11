import argparse
import glob
import os
import time

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from sort import Sort

np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument('--seq_path', help='Path to detections.', type=str, default='data')
    parser.add_argument('--phase', help='Subdirectory in seq_path.', type=str, default='train')
    parser.add_argument('--max_age',
                        help='Maximum number of frames to keep alive a track without associated detections.',
                        type=int,
                        default=1)
    parser.add_argument('--min_hits',
                        help='Minimum number of associated detections before track is initialized.',
                        type=int,
                        default=3)
    parser.add_argument('--iou_threshold',
                        help='Minimum IOU for match.',
                        type=float,
                        default=0.3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    total_time = 0.0
    total_frames = 0
    colors = np.random.rand(32, 3)  # used only for display

    pattern = os.path.join(args.seq_path, args.phase, '*', 'det', 'det.txt')
    for seq_detections_filepath in glob.glob(pattern):
        seq_filename = seq_detections_filepath[pattern.find('*'):].split(os.path.sep)[0]
        seq_detections = np.loadtxt(seq_detections_filepath, delimiter=',')

        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)

        if not os.path.exists('output'):
            os.makedirs('output')

        with open(os.path.join('output', f'{seq_filename}.txt'), 'w') as out_file:
            print(f'Processing {seq_filename}')
            for frame in range(int(seq_detections[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                detections = seq_detections[seq_detections[:, 0] == frame, 2:7]
                detections[:, 2:4] += detections[:, 0:2]  # convert from [x1, y1, w, h] to [x1, y1, x2, y2]
                total_frames += 1

                if args.display:
                    if not os.path.exists('mot_benchmark'):
                        print('\n\tERROR: mot_benchmark link not found!')
                        print('\n\n\tCREATE a symbolic link to the MOT benchmark')
                        print('\nE.g.\n\n\t$ ln -s /path/to/MOT15 mot_benchmark\n\n')

                    plt.ion()
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111, aspect='equal')

                    img_path = os.path.join('mot_benchmark', args.phase, seq_filename, 'img1', f'{frame:06d}.jpg')
                    img = io.imread(img_path)
                    ax1.imshow(img)
                    plt.title(seq_filename + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(detections)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for t in trackers:
                    print(f'{frame},{t[4]},{t[0]:.2f},{t[1]:.2f},{t[2] - t[0]:.2f},{t[3] - t[1]:.2f}, 1, -1, -1, -1',
                          file=out_file)
                    if args.display:
                        t = t.astype(np.int32)
                        ax1.add_patch(
                            patches.Rectangle(
                                (t[0], t[1]),
                                t[2] - t[0],
                                t[3] - t[1],
                                fill=False,
                                linewidth=3,
                                edgecolor=colors[t[4] % 32, :],
                            )
                        )

                if args.display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print(f"Total Tracking took: {total_time:.3f} for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")

    if args.display:
        print("Note: to get real runtime results run without the option: --display")


if __name__ == "__main__":
    main()
