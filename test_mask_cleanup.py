"""Regressions for the thin structure, pinholes and specks seen in SAM masks."""

import unittest

import numpy as np

from mask_cleanup import clean_mask


class MaskCleanupTests(unittest.TestCase):
    def test_balanced_repairs_speck_and_pinhole_without_filling_real_gap(self):
        mask = np.zeros((220, 300), dtype=bool)
        mask[65:180, 45:255] = True
        # A second substantial visible piece can belong to an occluded object.
        mask[28:54, 218:251] = True
        # A thin structural support connected diagonally to the main region.
        bar = [(88 + n, 65 - n) for n in range(37)]
        for x, y in bar:
            mask[y, x] = True
        mask[15:17, 15:17] = True  # unprompted noise
        mask[190:192, 275:277] = True  # deliberately included fragment
        mask[92:94, 75:77] = False  # accidental pinhole
        mask[100:102, 80:82] = False  # deliberate background prompt
        mask[92:134, 135:170] = False  # real opening
        original = mask.copy()

        result = clean_mask(mask, positive=[(275, 190)], negative=[(80, 100)], strength="balanced")

        self.assertFalse(result.mask[15:17, 15:17].any())
        self.assertTrue(result.mask[92:94, 75:77].all())
        self.assertFalse(result.mask[100, 80])
        self.assertTrue(result.mask[190, 275])
        self.assertFalse(result.mask[109, 153])
        self.assertTrue(result.mask[40, 232])
        self.assertGreaterEqual(sum(bool(result.mask[y, x]) for x, y in bar), 35)
        np.testing.assert_array_equal(mask, original)
        self.assertEqual(result.changed_pixels, np.count_nonzero(result.mask != original))

    def test_boundary_repair_preserves_more_than_ninety_nine_percent_of_large_mask(self):
        mask = np.zeros((220, 300), dtype=bool)
        mask[40:180, 40:260] = True
        mask[39, 50:250:2] = True  # rapid one-pixel alternation along the edge
        mask[180, 50:250:2] = True

        result = clean_mask(mask, strength="balanced").mask

        def perimeter(bits):
            frame = np.pad(bits, 1)
            return np.count_nonzero(np.diff(frame, axis=0)) + np.count_nonzero(np.diff(frame, axis=1))

        self.assertLess(perimeter(result), perimeter(mask) * 0.9)
        self.assertGreater(np.count_nonzero(result & mask) / np.count_nonzero(result | mask), 0.99)

    def test_off_preserves_binary_mask_exactly(self):
        mask = np.zeros((120, 150), dtype=np.uint8)
        mask[20:100, 20:130] = 255
        mask[46, 52] = 0
        mask[10, 10] = 255
        result = clean_mask(mask, strength="off")
        np.testing.assert_array_equal(result.mask, mask.astype(bool))
        self.assertEqual(result.changed_pixels, 0)


if __name__ == "__main__":
    unittest.main()
