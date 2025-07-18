import os
import logging
import pathlib
import schedule
from time import time
from datetime import timedelta
from timeloop import Timeloop
from monailabel.config import settings
from monailabel.utils.others.generic import (
    remove_file,
)

logger = logging.getLogger(__name__)


class ImageCache:
    """
    ImageCache manages a temporary cache directory for storing images,
    with automatic cleanup of expired entries.

    Attributes:
        cache_path (str): Filesystem path to the cache directory.
        cached_dirs (dict): Dictionary mapping file paths to expiry timestamps.
        cache_expiry_sec (int): Duration (in seconds) after which cache entries expire.
    """
    def __init__(self):
        """
        Initialize the ImageCache object.

        - Sets the cache directory path (from MONAI_LABEL settings or default).
        - Clears the existing cache directory.
        - Creates the cache directory if it does not exist.
        - Initializes internal structures.
        """
        cache_path = settings.MONAI_LABEL_DATASTORE_CACHE_PATH
        self.cache_path = (
            os.path.join(cache_path, "sam2")
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "sam2")
        )

        self.cached_dirs = {}
        self.cache_expiry_sec = 10 * 60 # Default: 10 minutes

        remove_file(self.cache_path)
        os.makedirs(self.cache_path, exist_ok=True)
        logger.info(f"Image Cache Initialized: {self.cache_path}")

    def cleanup(self):
        """
        Remove expired cache entries based on the current timestamp.

        For each entry in the cache, if its expiry time is earlier than now,
        the file is removed and the entry is deleted from the dictionary.
        """
        ts = time()
        expired = {k: v for k, v in self.cached_dirs.items() if v < ts}
        for k, v in expired.items():
            self.cached_dirs.pop(k)
            logger.info(f"Remove Expired Image: {k}; ExpiryTs: {v}; CurrentTs: {ts}")
            remove_file(k)

    def monitor(self):
        """
        Start a background thread that periodically (every 60 seconds)
        runs the cleanup function to delete expired cache entries.

        Combines the Timeloop and schedule libraries to handle periodic execution.
        """
        self.cleanup()
        time_loop = Timeloop()
        schedule.every(1).minutes.do(self.cleanup)

        @time_loop.job(interval=timedelta(seconds=60))
        def run_scheduler():
            schedule.run_pending()

        time_loop.start(block=False)#
