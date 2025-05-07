import unittest
import os
import pandas as pd
import numpy as np
from omnibin.metrics import generate_binary_classification_report

class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        cls.y_true = np.random.binomial(1, 0.3, n_samples)
        cls.y_scores = np.random.beta(2, 5, n_samples)
        
        # Adjust scores to have some correlation with true labels
        cls.y_scores[cls.y_true == 1] = np.random.beta(5, 2, sum(cls.y_true == 1))
        
        # Create test output directory
        cls.test_output_dir = "test_outputs"
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_report_generation(self):
        """Test the main report generation function"""
        output_path = os.path.join(self.test_output_dir, "test_report.pdf")
        
        # Generate report
        result_path = generate_binary_classification_report(
            y_true=self.y_true,
            y_scores=self.y_scores,
            output_path=output_path,
            n_bootstrap=100  # Use smaller number for testing
        )
        
        # Test that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.getsize(result_path) > 0)
        
        # Clean up
        os.remove(result_path)

    def test_input_validation(self):
        """Test input validation"""
        # Test with invalid y_true values
        with self.assertRaises(ValueError):
            generate_binary_classification_report(
                y_true=np.array([0, 1, 2]),  # Invalid label
                y_scores=np.array([0.1, 0.5, 0.9])
            )
        
        # Test with invalid y_scores values
        with self.assertRaises(ValueError):
            generate_binary_classification_report(
                y_true=np.array([0, 1, 0]),
                y_scores=np.array([-0.1, 1.5, 0.9])  # Values outside [0,1]
            )
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            generate_binary_classification_report(
                y_true=np.array([0, 1]),
                y_scores=np.array([0.1, 0.5, 0.9])
            )

    def test_bootstrap_consistency(self):
        """Test that bootstrap results are consistent"""
        output_path1 = os.path.join(self.test_output_dir, "test_report1.pdf")
        output_path2 = os.path.join(self.test_output_dir, "test_report2.pdf")
        
        # Generate two reports with same data and seed
        np.random.seed(42)
        generate_binary_classification_report(
            y_true=self.y_true,
            y_scores=self.y_scores,
            output_path=output_path1,
            n_bootstrap=100
        )
        
        np.random.seed(42)
        generate_binary_classification_report(
            y_true=self.y_true,
            y_scores=self.y_scores,
            output_path=output_path2,
            n_bootstrap=100
        )
        
        # Compare file sizes (they should be similar)
        size1 = os.path.getsize(output_path1)
        size2 = os.path.getsize(output_path2)
        self.assertAlmostEqual(size1, size2, delta=1000)  # Allow small differences due to PDF compression
        
        # Clean up
        os.remove(output_path1)
        os.remove(output_path2)

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with all positive labels
        output_path = os.path.join(self.test_output_dir, "all_positive.pdf")
        generate_binary_classification_report(
            y_true=np.ones(100),
            y_scores=np.random.random(100),
            output_path=output_path,
            n_bootstrap=100
        )
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)
        
        # Test with all negative labels
        output_path = os.path.join(self.test_output_dir, "all_negative.pdf")
        generate_binary_classification_report(
            y_true=np.zeros(100),
            y_scores=np.random.random(100),
            output_path=output_path,
            n_bootstrap=100
        )
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)
        
        # Test with perfect predictions
        output_path = os.path.join(self.test_output_dir, "perfect.pdf")
        generate_binary_classification_report(
            y_true=np.array([0, 1, 0, 1]),
            y_scores=np.array([0.1, 0.9, 0.2, 0.8]),
            output_path=output_path,
            n_bootstrap=100
        )
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test outputs"""
        if os.path.exists(cls.test_output_dir):
            for file in os.listdir(cls.test_output_dir):
                os.remove(os.path.join(cls.test_output_dir, file))
            os.rmdir(cls.test_output_dir)

if __name__ == '__main__':
    unittest.main() 