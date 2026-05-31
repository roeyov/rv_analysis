"""Shared logger instance + file/stdout setup for the bias_grid pipeline.

Every sub-module imports ``logger`` from here so a single
``setup_logging(output_dir)`` call at process start wires up both the
file and stdout handlers for the entire package.
"""

import os
import logging


logger = logging.getLogger("bias_grid")


def setup_logging(output_dir, level=logging.DEBUG):
    """Configure file + stdout logging. Call once in main()."""
    os.makedirs(output_dir, exist_ok=True)
    logger.setLevel(level)
    # File handler (DEBUG level — everything)
    fh = logging.FileHandler(
        os.path.join(output_dir, "bias_grid.log"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    # Stdout handler (INFO level)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(sh)
