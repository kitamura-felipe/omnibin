import unittest
import os
import numpy as np
import warnings
from omnibin.detection_metrics import (
    generate_detection_report,
    generate_lesion_detection_report,
    validate_detection_inputs
)
from omnibin.detection_utils import (
    DetectionColorScheme, calculate_iou, match_detections_to_ground_truth,
    calculate_precision_recall_curve, calculate_ap, calculate_froc_curve,
    calculate_froc_score, calculate_detection_metrics
)


class TestDetectionMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        np.random.seed(42)

        # Create synthetic detection data for 10 images
        cls.predictions = []
        cls.ground_truths = []

        for i in range(10):
            n_gt = np.random.randint(1, 5)
            n_pred = np.random.randint(0, 6)

            # Ground truths
            gts = []
            for _ in range(n_gt):
                x1 = np.random.randint(0, 400)
                y1 = np.random.randint(0, 400)
                w = np.random.randint(20, 100)
                h = np.random.randint(20, 100)
                gts.append([x1, y1, x1 + w, y1 + h])
            cls.ground_truths.append(gts)

            # Predictions (some overlapping with GT, some not)
            preds = []
            for j in range(n_pred):
                if j < len(gts) and np.random.random() > 0.3:
                    # Create prediction overlapping with GT
                    gt = gts[j]
                    offset = np.random.randint(-10, 10, 4)
                    box = [
                        max(0, gt[0] + offset[0]),
                        max(0, gt[1] + offset[1]),
                        gt[2] + offset[2],
                        gt[3] + offset[3]
                    ]
                else:
                    # Random prediction
                    x1 = np.random.randint(0, 400)
                    y1 = np.random.randint(0, 400)
                    w = np.random.randint(20, 100)
                    h = np.random.randint(20, 100)
                    box = [x1, y1, x1 + w, y1 + h]

                preds.append({
                    'box': box,
                    'score': np.random.uniform(0.3, 1.0)
                })
            cls.predictions.append(preds)

        # Create test output directory
        cls.test_output_dir = "test_outputs_detection"
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_iou_calculation(self):
        """Test IoU calculation"""
        # Perfect overlap
        box = [0, 0, 10, 10]
        self.assertAlmostEqual(calculate_iou(box, box), 1.0, places=5)

        # No overlap
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        self.assertEqual(calculate_iou(box1, box2), 0.0)

        # 50% overlap
        box1 = [0, 0, 10, 10]
        box2 = [5, 0, 15, 10]
        # Intersection = 5*10 = 50, Union = 10*10 + 10*10 - 50 = 150
        self.assertAlmostEqual(calculate_iou(box1, box2), 50 / 150, places=5)

    def test_matching(self):
        """Test detection matching"""
        predictions = [
            {'box': [0, 0, 10, 10], 'score': 0.9},
            {'box': [20, 20, 30, 30], 'score': 0.8}
        ]
        ground_truths = [
            [0, 0, 10, 10],
            [50, 50, 60, 60]
        ]

        tp, fp, fn, _, _ = match_detections_to_ground_truth(predictions, ground_truths, iou_threshold=0.5)

        self.assertEqual(tp, 1)  # First prediction matches first GT
        self.assertEqual(fp, 1)  # Second prediction doesn't match
        self.assertEqual(fn, 1)  # Second GT not detected

    def test_ap_calculation(self):
        """Test Average Precision calculation"""
        # Perfect predictions
        precisions = np.array([1.0, 1.0, 1.0])
        recalls = np.array([0.33, 0.66, 1.0])
        ap = calculate_ap(precisions, recalls)
        self.assertAlmostEqual(ap, 1.0, places=2)

    def test_froc_score(self):
        """Test FROC score calculation"""
        sensitivities = np.array([0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
        fp_per_image = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0])

        score = calculate_froc_score(sensitivities, fp_per_image)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)

    def test_report_generation(self):
        """Test the main report generation function"""
        output_path = os.path.join(self.test_output_dir, "test_detection_report.pdf")

        result_path = generate_detection_report(
            predictions=self.predictions,
            ground_truths=self.ground_truths,
            output_path=output_path,
            n_bootstrap=30
        )

        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)
        os.remove(result_path)

    def test_lesion_detection_report(self):
        """Test lesion detection report generation"""
        output_path = os.path.join(self.test_output_dir, "test_lesion_detection.pdf")

        result_path = generate_lesion_detection_report(
            predictions=self.predictions,
            ground_truths=self.ground_truths,
            output_path=output_path,
            n_bootstrap=30
        )

        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)
        os.remove(result_path)

    def test_input_validation(self):
        """Test input validation"""
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            validate_detection_inputs(
                predictions=[[], []],
                ground_truths=[[]]
            )

        # Test with empty input
        with self.assertRaises(ValueError):
            validate_detection_inputs(
                predictions=[],
                ground_truths=[]
            )

        # Test with invalid prediction format
        with self.assertRaises(ValueError):
            validate_detection_inputs(
                predictions=[[{'score': 0.9}]],  # Missing box
                ground_truths=[[[0, 0, 10, 10]]]
            )

        # Test with invalid box format
        with self.assertRaises(ValueError):
            validate_detection_inputs(
                predictions=[[{'box': [0, 0, 10], 'score': 0.9}]],  # Only 3 values
                ground_truths=[[[0, 0, 10, 10]]]
            )

    def test_empty_predictions(self):
        """Test with no predictions"""
        output_path = os.path.join(self.test_output_dir, "no_predictions.pdf")

        # All images have no predictions
        empty_preds = [[] for _ in range(len(self.ground_truths))]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_path = generate_detection_report(
                predictions=empty_preds,
                ground_truths=self.ground_truths,
                output_path=output_path,
                n_bootstrap=20
            )

        self.assertTrue(os.path.exists(result_path))
        os.remove(result_path)

    def test_perfect_detection(self):
        """Test with perfect detection"""
        output_path = os.path.join(self.test_output_dir, "perfect_detection.pdf")

        # Create perfect predictions
        perfect_preds = []
        for gts in self.ground_truths:
            preds = [{'box': gt, 'score': 1.0} for gt in gts]
            perfect_preds.append(preds)

        result_path = generate_detection_report(
            predictions=perfect_preds,
            ground_truths=self.ground_truths,
            output_path=output_path,
            n_bootstrap=20
        )

        self.assertTrue(os.path.exists(result_path))
        os.remove(result_path)

    def test_color_schemes(self):
        """Test different color schemes"""
        for scheme in [DetectionColorScheme.DEFAULT, DetectionColorScheme.MONOCHROME,
                       DetectionColorScheme.VIBRANT]:
            output_path = os.path.join(self.test_output_dir, f"test_{scheme.name}.pdf")

            result_path = generate_detection_report(
                predictions=self.predictions[:3],
                ground_truths=self.ground_truths[:3],
                output_path=output_path,
                n_bootstrap=20,
                color_scheme=scheme
            )

            self.assertTrue(os.path.exists(result_path))
            os.remove(result_path)

    def test_detection_metrics_calculation(self):
        """Test comprehensive metrics calculation"""
        metrics = calculate_detection_metrics(self.predictions, self.ground_truths)

        # Check all expected metrics are present
        expected_metrics = ['AP@50', 'AP@75', 'mAP', 'FROC Score',
                           'Precision@50', 'Recall@50', 'F1@50',
                           'Detection Rate', 'Avg FP/Image']

        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

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
