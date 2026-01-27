import unittest
import os
import numpy as np
import warnings
from omnibin.regression_metrics import generate_regression_report, validate_regression_inputs
from omnibin.regression_utils import RegressionColorScheme


class TestRegressionMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        np.random.seed(42)
        n_samples = 200

        # Create synthetic regression data with some correlation
        cls.y_true = np.random.randn(n_samples) * 10 + 50
        cls.y_pred = cls.y_true + np.random.randn(n_samples) * 3

        # Create test output directory
        cls.test_output_dir = "test_outputs_regression"
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_report_generation(self):
        """Test the main report generation function"""
        output_path = os.path.join(self.test_output_dir, "test_regression_report.pdf")

        # Generate report
        result_path = generate_regression_report(
            y_true=self.y_true,
            y_pred=self.y_pred,
            output_path=output_path,
            n_bootstrap=50  # Use smaller number for testing
        )

        # Test that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)

        # Clean up
        os.remove(result_path)

    def test_input_validation(self):
        """Test input validation"""
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            validate_regression_inputs(
                y_true=np.array([1, 2, 3]),
                y_pred=np.array([1, 2])
            )

        # Test with too few samples
        with self.assertRaises(ValueError):
            validate_regression_inputs(
                y_true=np.array([1, 2]),
                y_pred=np.array([1, 2])
            )

        # Test with NaN values
        with self.assertRaises(ValueError):
            validate_regression_inputs(
                y_true=np.array([1, 2, np.nan, 4, 5]),
                y_pred=np.array([1, 2, 3, 4, 5])
            )

        # Test with infinite values
        with self.assertRaises(ValueError):
            validate_regression_inputs(
                y_true=np.array([1, 2, np.inf, 4, 5]),
                y_pred=np.array([1, 2, 3, 4, 5])
            )

    def test_color_schemes(self):
        """Test different color schemes"""
        for scheme in [RegressionColorScheme.DEFAULT, RegressionColorScheme.MONOCHROME,
                       RegressionColorScheme.VIBRANT]:
            output_path = os.path.join(self.test_output_dir, f"test_{scheme.name}.pdf")

            result_path = generate_regression_report(
                y_true=self.y_true[:50],
                y_pred=self.y_pred[:50],
                output_path=output_path,
                n_bootstrap=20,
                color_scheme=scheme
            )

            self.assertTrue(os.path.exists(result_path))
            os.remove(result_path)

    def test_perfect_predictions(self):
        """Test with perfect predictions"""
        output_path = os.path.join(self.test_output_dir, "perfect_regression.pdf")

        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = y_true.copy()

        result_path = generate_regression_report(
            y_true=y_true,
            y_pred=y_pred,
            output_path=output_path,
            n_bootstrap=20
        )

        self.assertTrue(os.path.exists(result_path))
        os.remove(result_path)

    def test_negative_values(self):
        """Test with negative values"""
        output_path = os.path.join(self.test_output_dir, "negative_regression.pdf")

        y_true = np.random.randn(50) * 100
        y_pred = y_true + np.random.randn(50) * 10

        result_path = generate_regression_report(
            y_true=y_true,
            y_pred=y_pred,
            output_path=output_path,
            n_bootstrap=20
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
