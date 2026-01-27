import unittest
import os
import numpy as np
import warnings
from omnibin.segmentation_metrics import (
    generate_segmentation_report,
    generate_multiclass_segmentation_report,
    validate_segmentation_inputs
)
from omnibin.segmentation_utils import (
    SegmentationColorScheme, dice_score, iou_score, pixel_accuracy,
    sensitivity, specificity, hausdorff_distance, average_surface_distance
)


class TestSegmentationMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        np.random.seed(42)

        # Create synthetic 2D segmentation masks
        cls.y_true_2d = np.zeros((256, 256), dtype=np.uint8)
        cls.y_true_2d[80:180, 80:180] = 1  # Square region

        cls.y_pred_2d = np.zeros((256, 256), dtype=np.uint8)
        cls.y_pred_2d[90:190, 90:190] = 1  # Slightly shifted square

        # Create synthetic 3D segmentation masks
        cls.y_true_3d = np.zeros((20, 128, 128), dtype=np.uint8)
        cls.y_true_3d[5:15, 40:90, 40:90] = 1

        cls.y_pred_3d = np.zeros((20, 128, 128), dtype=np.uint8)
        cls.y_pred_3d[5:15, 45:95, 45:95] = 1

        # Create test output directory
        cls.test_output_dir = "test_outputs_segmentation"
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_dice_score(self):
        """Test Dice score calculation"""
        # Perfect match
        mask = np.ones((10, 10))
        self.assertAlmostEqual(dice_score(mask, mask), 1.0, places=5)

        # No overlap
        mask1 = np.zeros((10, 10))
        mask1[:5, :] = 1
        mask2 = np.zeros((10, 10))
        mask2[5:, :] = 1
        self.assertLess(dice_score(mask1, mask2), 0.01)

    def test_iou_score(self):
        """Test IoU score calculation"""
        # Perfect match
        mask = np.ones((10, 10))
        self.assertAlmostEqual(iou_score(mask, mask), 1.0, places=5)

        # 50% overlap
        mask1 = np.zeros((10, 10))
        mask1[:, :5] = 1
        mask2 = np.zeros((10, 10))
        mask2[:, 2:7] = 1
        # IoU should be 3/7 = 0.428...
        self.assertAlmostEqual(iou_score(mask1, mask2), 3 / 7, places=2)

    def test_pixel_accuracy(self):
        """Test pixel accuracy calculation"""
        mask1 = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
        mask2 = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        # 7/8 = 0.875
        self.assertAlmostEqual(pixel_accuracy(mask1, mask2), 7 / 8, places=5)

    def test_report_generation_2d(self):
        """Test report generation for 2D masks"""
        output_path = os.path.join(self.test_output_dir, "test_segmentation_2d.pdf")

        result_path = generate_segmentation_report(
            y_true=self.y_true_2d,
            y_pred=self.y_pred_2d,
            output_path=output_path,
            n_bootstrap=30
        )

        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)
        os.remove(result_path)

    def test_report_generation_3d(self):
        """Test report generation for 3D masks"""
        output_path = os.path.join(self.test_output_dir, "test_segmentation_3d.pdf")

        result_path = generate_segmentation_report(
            y_true=self.y_true_3d,
            y_pred=self.y_pred_3d,
            output_path=output_path,
            n_bootstrap=30
        )

        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)
        os.remove(result_path)

    def test_input_validation(self):
        """Test input validation"""
        # Test with shape mismatch
        with self.assertRaises(ValueError):
            validate_segmentation_inputs(
                y_true=np.zeros((10, 10)),
                y_pred=np.zeros((10, 20))
            )

        # Test with 1D input
        with self.assertRaises(ValueError):
            validate_segmentation_inputs(
                y_true=np.zeros(10),
                y_pred=np.zeros(10)
            )

        # Test with 4D input
        with self.assertRaises(ValueError):
            validate_segmentation_inputs(
                y_true=np.zeros((2, 10, 10, 10)),
                y_pred=np.zeros((2, 10, 10, 10))
            )

    def test_perfect_segmentation(self):
        """Test with perfect segmentation"""
        output_path = os.path.join(self.test_output_dir, "perfect_segmentation.pdf")

        result_path = generate_segmentation_report(
            y_true=self.y_true_2d,
            y_pred=self.y_true_2d.copy(),
            output_path=output_path,
            n_bootstrap=20
        )

        self.assertTrue(os.path.exists(result_path))
        os.remove(result_path)

    def test_empty_masks(self):
        """Test with empty masks"""
        output_path = os.path.join(self.test_output_dir, "empty_segmentation.pdf")

        y_true = np.zeros((100, 100), dtype=np.uint8)
        y_pred = np.zeros((100, 100), dtype=np.uint8)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_path = generate_segmentation_report(
                y_true=y_true,
                y_pred=y_pred,
                output_path=output_path,
                n_bootstrap=20
            )

        self.assertTrue(os.path.exists(result_path))
        os.remove(result_path)

    def test_color_schemes(self):
        """Test different color schemes"""
        for scheme in [SegmentationColorScheme.DEFAULT, SegmentationColorScheme.MONOCHROME,
                       SegmentationColorScheme.VIBRANT]:
            output_path = os.path.join(self.test_output_dir, f"test_{scheme.name}.pdf")

            result_path = generate_segmentation_report(
                y_true=self.y_true_2d,
                y_pred=self.y_pred_2d,
                output_path=output_path,
                n_bootstrap=20,
                color_scheme=scheme
            )

            self.assertTrue(os.path.exists(result_path))
            os.remove(result_path)

    def test_multiclass_segmentation(self):
        """Test multi-class segmentation report"""
        output_path = os.path.join(self.test_output_dir, "multiclass_segmentation.pdf")

        # Create multi-class masks
        y_true = np.zeros((128, 128), dtype=np.uint8)
        y_true[20:50, 20:50] = 1
        y_true[60:90, 60:90] = 2
        y_true[20:50, 60:90] = 3

        y_pred = np.zeros((128, 128), dtype=np.uint8)
        y_pred[22:52, 22:52] = 1
        y_pred[58:88, 58:88] = 2
        y_pred[22:52, 62:92] = 3

        result_path = generate_multiclass_segmentation_report(
            y_true=y_true,
            y_pred=y_pred,
            output_path=output_path,
            n_bootstrap=20,
            class_names={1: "Tumor", 2: "Edema", 3: "Necrosis"}
        )

        self.assertTrue(os.path.exists(result_path))
        os.remove(result_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test outputs"""
        if os.path.exists(cls.test_output_dir):
            for file in os.listdir(cls.test_output_dir):
                file_path = os.path.join(cls.test_output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    for subfile in os.listdir(file_path):
                        os.remove(os.path.join(file_path, subfile))
                    os.rmdir(file_path)
            os.rmdir(cls.test_output_dir)


if __name__ == '__main__':
    unittest.main()
