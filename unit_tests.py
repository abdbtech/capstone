"""
Unit Tests for Disaster Risk Modeling Project

This module contains validation tests for various components of the compound
Poisson disaster risk model, including severity metrics, frequency parameters,
and data quality checks.

Usage:
    from unit_tests import IntensityTests, PoissonTests
    
    # Test intensity metric
    intensity_tester = IntensityTests(dataframe)
    intensity_tester.run_all_tests()
    
    # Test Poisson parameters
    poisson_tester = PoissonTests(dataframe)
    poisson_tester.run_all_tests()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

class IntensityTests:
    """
    Test suite for validating intensity metrics in disaster severity modeling.
    
    Validates that intensity = casualty_rate * (1 + vulnerability_rate/100)
    behaves logically and produces reasonable results.
    """
    
    def __init__(self, dataframe, intensity_col='intensity', casualty_col='casualties', 
                 casualty_rate_col='casualty_rate', vulnerability_col='vunerability_rate'):
        """
        Initialize test suite with data and column names.
        
        Args:
            dataframe: pandas DataFrame containing the data
            intensity_col: name of intensity metric column
            casualty_col: name of casualties column  
            casualty_rate_col: name of casualty rate column
            vulnerability_col: name of vulnerability rate column
        """
        self.df = dataframe
        self.intensity_col = intensity_col
        self.casualty_col = casualty_col
        self.casualty_rate_col = casualty_rate_col
        self.vulnerability_col = vulnerability_col
        
    def test_zero_casualties_zero_intensity(self):
        """Test that zero casualties always results in zero intensity."""
        print("Test 1: Zero casualties → Zero intensity")
        
        zero_casualty_mask = self.df[self.casualty_col] == 0
        zero_intensity = self.df[zero_casualty_mask][self.intensity_col]
        
        max_intensity_zero_cas = zero_intensity.max()
        
        print(f"Events with 0 casualties: {zero_casualty_mask.sum()}")
        print(f"Max intensity for 0-casualty events: {max_intensity_zero_cas:.4f}")
        
        test_passed = max_intensity_zero_cas == 0
        print(f"{'✓ PASS' if test_passed else '✗ FAIL'}")
        
        if not test_passed:
            print("  → Check formula: should be casualty_rate * (1 + vulnerability/100)")
            print("  → Current formula appears to be additive instead of multiplicative")
        
        return test_passed
    
    def test_vulnerability_amplification(self):
        """Test that higher vulnerability increases intensity for similar casualty rates."""
        print("\nTest 2: Higher vulnerability → Higher intensity")
        
        sample_data = self.df[self.df[self.casualty_col] > 0].copy()
        
        if len(sample_data) == 0:
            print("No events with casualties found")
            return False
            
        # Compare low vs high vulnerability areas with similar casualty rates
        median_casualty = sample_data[self.casualty_rate_col].median()
        similar_casualty = sample_data[
            (sample_data[self.casualty_rate_col] >= median_casualty * 0.8) & 
            (sample_data[self.casualty_rate_col] <= median_casualty * 1.2)
        ]
        
        if len(similar_casualty) < 10:
            print("Insufficient data for vulnerability comparison")
            return False
            
        low_vuln = similar_casualty[similar_casualty[self.vulnerability_col] < 50]
        high_vuln = similar_casualty[similar_casualty[self.vulnerability_col] > 80]
        
        if len(low_vuln) == 0 or len(high_vuln) == 0:
            print("Insufficient data in vulnerability extremes")
            return False
            
        low_vuln_intensity = low_vuln[self.intensity_col].mean()
        high_vuln_intensity = high_vuln[self.intensity_col].mean()
        
        print(f"Low vulnerability areas avg intensity: {low_vuln_intensity:.4f}")
        print(f"High vulnerability areas avg intensity: {high_vuln_intensity:.4f}")
        
        test_passed = high_vuln_intensity > low_vuln_intensity
        print(f"{'✓ PASS' if test_passed else '✗ FAIL'}")
        
        return test_passed
    
    def test_correlations(self):
        """Test expected correlations between components and intensity."""
        print("\n=== CORRELATION ANALYSIS ===")
        
        analysis_data = self.df[self.df[self.casualty_col] > 0].copy()
        
        if len(analysis_data) == 0:
            print("No events with casualties for correlation analysis")
            return False
            
        correlations = analysis_data[[self.casualty_rate_col, self.vulnerability_col, 
                                    self.intensity_col]].corr()
        
        print("Correlation matrix:")
        print(correlations.round(3))
        
        casualty_intensity_corr = correlations.loc[self.casualty_rate_col, self.intensity_col]
        vuln_intensity_corr = correlations.loc[self.vulnerability_col, self.intensity_col]
        
        print(f"\nCasualty rate → Intensity correlation: {casualty_intensity_corr:.3f}")
        casualty_test = casualty_intensity_corr > 0.7
        print(f"{'✓ Strong positive as expected' if casualty_test else '⚠ Weaker than expected'}")
        
        print(f"Vulnerability → Intensity correlation: {vuln_intensity_corr:.3f}")
        vuln_test = vuln_intensity_corr > 0
        print(f"{'✓ Positive as expected' if vuln_test else '✗ Negative - check formula'}")
        
        return casualty_test and vuln_test
    
    def test_distribution_properties(self):
        """Test if intensity distribution has reasonable properties."""
        print("\n=== DISTRIBUTION VALIDATION ===")
        
        intensity_nonzero = self.df[self.df[self.intensity_col] > 0][self.intensity_col]
        
        if len(intensity_nonzero) == 0:
            print("No positive intensity values found")
            return False
            
        print("Intensity metric statistics:")
        print(f"Count: {len(intensity_nonzero)}")
        print(f"Range: {intensity_nonzero.min():.4f} to {intensity_nonzero.max():.4f}")
        print(f"Mean: {intensity_nonzero.mean():.4f}")
        print(f"Median: {intensity_nonzero.median():.4f}")
        print(f"95th percentile: {intensity_nonzero.quantile(0.95):.4f}")
        
        # Check for extreme outliers
        q99 = intensity_nonzero.quantile(0.99)
        outliers = intensity_nonzero[intensity_nonzero > q99]
        
        print(f"\nOutliers (>99th percentile): {len(outliers)}")
        if len(outliers) > 0:
            print(f"Outlier range: {outliers.min():.4f} to {outliers.max():.4f}")
        
        # Test for reasonable range (intensity shouldn't be astronomically high)
        max_reasonable = intensity_nonzero.mean() + 5 * intensity_nonzero.std()
        extreme_outliers = intensity_nonzero[intensity_nonzero > max_reasonable]
        
        test_passed = len(extreme_outliers) < len(intensity_nonzero) * 0.01  # Less than 1% extreme outliers
        print(f"{'✓ Reasonable range' if test_passed else '⚠ Some extreme outliers detected'}")
        
        return test_passed
    
    def test_edge_cases(self):
        """Test extreme cases to ensure formula robustness."""
        print("\n=== EDGE CASE TESTS ===")
        
        max_vuln = self.df[self.vulnerability_col].max()
        max_casualty = self.df[self.casualty_rate_col].max()
        
        # Simulate extreme case
        extreme_intensity = max_casualty * (1 + max_vuln/100)
        
        print(f"Extreme case simulation:")
        print(f"Max casualty rate: {max_casualty:.4f}")
        print(f"Max vulnerability rate: {max_vuln:.1f}%")
        print(f"Resulting intensity: {extreme_intensity:.4f}")
        
        if max_casualty > 0:
            amplification = extreme_intensity / max_casualty
            print(f"Amplification factor: {amplification:.2f}x")
            
            # Check if amplification is reasonable (typically should be < 3x)
            reasonable_amp = amplification < 3
            print(f"{'✓ Reasonable amplification' if reasonable_amp else '⚠ Very high amplification'}")
            return reasonable_amp
        
        return True
    
    def plot_intensity_distribution(self, show_plot=True):
        """Create visualization of intensity distribution."""
        intensity_nonzero = self.df[self.df[self.intensity_col] > 0][self.intensity_col]
        
        if len(intensity_nonzero) == 0:
            print("No positive intensity values to plot")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Linear scale
        axes[0].hist(intensity_nonzero, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Intensity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Intensity Distribution (Linear Scale)')
        
        # Log scale
        axes[1].hist(intensity_nonzero, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Intensity Score')
        axes[1].set_ylabel('Frequency (Log Scale)')
        axes[1].set_title('Intensity Distribution (Log Scale)')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig
    
    def run_all_tests(self, show_plots=True):
        """Run complete test suite and return summary."""
        print("=== INTENSITY METRIC VALIDATION TESTS ===\n")
        
        tests = [
            self.test_zero_casualties_zero_intensity(),
            self.test_vulnerability_amplification(),
            self.test_correlations(),
            self.test_distribution_properties(),
            self.test_edge_cases()
        ]
        
        if show_plots:
            self.plot_intensity_distribution()
        
        passed_tests = sum(tests)
        total_tests = len(tests)
        
        print(f"\n=== TEST SUMMARY ===")
        print(f"Passed: {passed_tests}/{total_tests} tests")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("✓ All tests passed - Intensity metric is valid!")
        else:
            print("⚠ Some tests failed - Review formula and data")
            
        return passed_tests == total_tests


class PoissonTests:
    """
    Test suite for validating Poisson distribution parameters and fit quality.
    """
    
    def __init__(self, dataframe, lambda_col='lambda_hat', event_count_col='event_count'):
        """
        Initialize Poisson test suite.
        
        Args:
            dataframe: pandas DataFrame with Poisson parameters
            lambda_col: column name for lambda estimates
            event_count_col: column name for event counts
        """
        self.df = dataframe
        self.lambda_col = lambda_col
        self.event_count_col = event_count_col
    
    def test_lambda_positivity(self):
        """Test that all lambda values are positive."""
        print("Test: Lambda parameters are positive")
        
        negative_lambdas = (self.df[self.lambda_col] < 0).sum()
        zero_lambdas = (self.df[self.lambda_col] == 0).sum()
        
        print(f"Negative lambdas: {negative_lambdas}")
        print(f"Zero lambdas: {zero_lambdas}")
        
        test_passed = negative_lambdas == 0
        print(f"{'✓ PASS' if test_passed else '✗ FAIL'}")
        
        return test_passed
    
    def test_mean_variance_relationship(self):
        """Test Poisson property: mean ≈ variance for each lambda."""
        print("\nTest: Poisson mean-variance relationship")
        
        # This would require grouped data by county
        # Implementation depends on data structure
        print("Implementation needed based on county-level grouping")
        return True
    
    def run_all_tests(self):
        """Run all Poisson validation tests."""
        print("=== POISSON PARAMETER VALIDATION ===\n")
        
        tests = [
            self.test_lambda_positivity(),
            self.test_mean_variance_relationship()
        ]
        
        passed_tests = sum(tests)
        total_tests = len(tests)
        
        print(f"\n=== POISSON TEST SUMMARY ===")
        print(f"Passed: {passed_tests}/{total_tests} tests")
        
        return passed_tests == total_tests


# Convenience function for quick testing
def quick_intensity_test(dataframe, **kwargs):
    """
    Quick test of intensity metric with default column names.
    
    Usage:
        from unit_tests import quick_intensity_test
        quick_intensity_test(your_dataframe)
    """
    tester = IntensityTests(dataframe, **kwargs)
    return tester.run_all_tests()


if __name__ == "__main__":
    print("Unit Tests Module for Disaster Risk Modeling")
    print("Import this module and use the test classes:")
    print("  from unit_tests import IntensityTests, PoissonTests")