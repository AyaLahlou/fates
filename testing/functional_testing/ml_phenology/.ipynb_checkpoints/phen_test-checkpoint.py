"""
Concrete class for running the quadtratic functional tests for FATES.
"""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_color_palette
from functional_class import FunctionalTest


class PhenologyTest(FunctionalTest):
    """Phenology test class
    """

    name = "phenology"

    def __init__(self, test_dict):
        super().__init__(
            PhenologyTest.name,
            test_dict["test_dir"],
            test_dict["test_exe"],
            test_dict["out_file"],
            test_dict["use_param_file"],
            test_dict["other_args"],
        )
        self.plot = True

    def plot_output(self, run_dir: str, save_figs: bool, plot_dir: str):
        """Reads in and plots quadratic formula test output

        Args:
            run_dir (str): run directory
            out_file (str): output file
            save_figs (bool): whether or not to save the figures
            plot_dir (str): plot directory
        """
        pass
