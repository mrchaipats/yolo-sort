import numpy as np
import numpy.ma as ma

from utils import associate_detections_to_trackers
from kalman import KalmanBoxTracker


class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections=np.empty((0, 5))):
        self.frame_count += 1

        tracks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, track in enumerate(tracks):
            position = self.trackers[t].predict()[0]
            tracks[:] = [position[0], position[1], position[2], position[3], 0]
            if np.any(np.isnan(position)):
                to_del.append(t)

        tracks = ma.compress_rows(ma.masked_invalid(tracks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_detections, unmatched_tracks = associate_detections_to_trackers(
            detections, tracks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :])

        # create and initialize new trackers for unmatched detections
        for i in unmatched_detections:
            new_track = KalmanBoxTracker(detections[i, :])
            self.trackers.append(new_track)

        i = len(self.trackers)
        for track in reversed(self.trackers):
            track_state = track.get_state()[0]
            if track.time_since_update < 1 and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((track_state, [track.id + 1])).reshape(1, -1))
            i -= 1

            # remove dead tracklet which has not been associated with any detection for more than max_age frames
            if track.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 5))
