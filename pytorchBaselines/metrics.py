import numpy as np
import scipy


class Metrics:
    def __init__(self, logging_obj):
        self._metrics_dict = dict()
        self.logging = logging_obj

    def add_metric(self, name: str, sample: list):
        self._metrics_dict[name] = self._calculate_metrics(sample)

    def _calculate_metrics(self, sample):
        def calculate_confidence_interval(sample, confidence_level=0.9):
            sample = np.array(sample)
            degrees_freedom = sample.size - 1
            sample_mean = np.mean(sample)
            sample_standard_error = scipy.stats.sem(sample)
            confidence_interval = scipy.stats.t.interval(
                confidence_level, degrees_freedom, sample_mean, sample_standard_error
            )
            # return tuple of lower and upper values
            return list(confidence_interval)

        return [np.mean(sample), np.std(sample), calculate_confidence_interval(sample)]

    def log_metrics(self, name="all"):
        def log_single_metric(key):
            self.logging.info("")
            self.logging.info(f"{key} ======")
            self.logging.info(f"MEAN: {self._metrics_dict[key][0]:.4f}")
            self.logging.info(f"STD DEV: {self._metrics_dict[key][1]:.4f}")
            self.logging.info(
                f"CI: [{self._metrics_dict[key][2][0]:.4f},{self._metrics_dict[key][2][1]:.4f}]"
            )
            return None

        if "all" in name:
            for key in self._metrics_dict.keys():
                log_single_metric(key)
        elif name in self._metrics_dict.keys():
            log_single_metric(name)
        else:
            raise KeyError(f"{name} not in metrics_dict")
