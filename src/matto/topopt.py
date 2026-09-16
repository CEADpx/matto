"""
Entry point used by the input files: ``from matto.topopt import topopt``.

The work is done by OptimizationDriver in driver.py; this keeps the
call the examples make.
"""

from .driver import OptimizationDriver


def topopt(problem):
    """Run the optimization problem described by ``problem``."""
    OptimizationDriver(problem).run()
