"""Checkpoint export rejects malformed detector output before scoring references."""

import unittest

from demo.auto_yolo_worker import checkpoint_detections


class CheckpointGeometryTests(unittest.TestCase):
    def test_invalid_predictions_are_reported_separately_from_detections(self):
        rows = [(0., [1., 2., 9., 10.], .6),
                (0., [10., 1., 10., 7.], .02),
                (0., [-4., 2., 2., 9.], .3),
                (2., [1., 1., 2., 2.], .7),
                (0., [float('nan'), 1., 2., 2.], .4)]
        detections, invalid = checkpoint_detections(rows, ['hammer'], 10, 10)
        self.assertEqual([item['xyxy'] for item in detections], [[1., 2., 9., 10.], [0., 2., 2., 9.]])
        self.assertEqual([item['reason'] for item in invalid],
                         ['zero_area_after_clipping', 'invalid_class', 'nonfinite_geometry'])
        self.assertIsNone(invalid[-1]['xyxy'][0])


if __name__ == '__main__':
    unittest.main()
