# statistical_analysis_enhanced.py

import numpy as np
import scipy.stats as stats
import logging
from typing import Tuple, List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StatisticalAnalysisEnhanced:
    """
    A class for performing advanced statistical analysis including Bayesian inference,
    advanced hypothesis testing, and uncertainty quantification.
    """

    def __init__(self):
        """
        Initializes the StatisticalAnalysisEnhanced class.
        """
        logging.info("StatisticalAnalysisEnhanced class initialized.")

    def bayesian_inference(self, data: np.ndarray, prior_mean: float, prior_std: float) -> Tuple[float, float]:
        """
        Perform Bayesian inference to update the mean and standard deviation of a normal distribution.

        Parameters:
        data (np.ndarray): Observed data.
        prior_mean (float): Prior mean of the distribution.
        prior_std (float): Prior standard deviation of the distribution.

        Returns:
        Tuple[float, float]: Posterior mean and standard deviation.
        """
        try:
            n = len(data)
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)

            posterior_variance = 1 / ((1 / prior_std**2) + (n / sample_std**2))
            posterior_mean = posterior_variance * ((prior_mean / prior_std**2) + (n * sample_mean / sample_std**2))
            posterior_std = np.sqrt(posterior_variance)

            logging.info("Bayesian inference completed.")
            return posterior_mean, posterior_std
        except Exception as e:
            logging.error(f"Error in bayesian_inference: {e}")
            raise

    def advanced_hypothesis_testing(self, data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform advanced hypothesis testing using a two-sample t-test.

        Parameters:
        data1 (np.ndarray): First sample data.
        data2 (np.ndarray): Second sample data.
        alpha (float): Significance level. Default is 0.05.

        Returns:
        Dict[str, float]: Test statistic and p-value.
        """
        try:
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            result = {
                't_statistic': t_stat,
                'p_value': p_value,
                'reject_null': p_value < alpha
            }
            logging.info("Advanced hypothesis testing completed.")
            return result
        except Exception as e:
            logging.error(f"Error in advanced_hypothesis_testing: {e}")
            raise

    def uncertainty_quantification(self, data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Quantify uncertainty by calculating the confidence interval for the mean of the data.

        Parameters:
        data (np.ndarray): Data for which to calculate the confidence interval.
        confidence_level (float): Confidence level for the interval. Default is 0.95.

        Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval.
        """
        try:
            n = len(data)
            mean = np.mean(data)
            sem = stats.sem(data)
            confidence_interval = stats.t.interval(confidence_level, n-1, loc=mean, scale=sem)

            logging.info("Uncertainty quantification completed.")
            return confidence_interval
        except Exception as e:
            logging.error(f"Error in uncertainty_quantification: {e}")
            raise

# Example usage
if __name__ == "__main__":
    analysis = StatisticalAnalysisEnhanced()
    
    # Bayesian Inference Example
    data = np.random.normal(10, 2, size=100)
    prior_mean = 8
    prior_std = 3
    posterior_mean, posterior_std = analysis.bayesian_inference(data, prior_mean, prior_std)
    print(f"Posterior Mean: {posterior_mean}, Posterior Std: {posterior_std}")

    # Advanced Hypothesis Testing Example
    data1 = np.random.normal(10, 2, size=100)
    data2 = np.random.normal(12, 2, size=100)
    test_results = analysis.advanced_hypothesis_testing(data1, data2)
    print(f"Test Statistic: {test_results['t_statistic']}, P-value: {test_results['p_value']}, Reject Null: {test_results['reject_null']}")

    # Uncertainty Quantification Example
    confidence_interval = analysis.uncertainty_quantification(data)
    print(f"Confidence Interval: {confidence_interval}")
```
