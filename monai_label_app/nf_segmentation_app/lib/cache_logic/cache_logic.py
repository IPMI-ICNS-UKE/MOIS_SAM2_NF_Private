import os
import logging
import pathlib
from datetime import timedelta
from time import time
import schedule
from timeloop import Timeloop
from monailabel.config import settings
from monailabel.utils.others.generic import (
    remove_file,
)

logger = logging.getLogger(__name__)

class ImageCache:
    def __init__(self):
        cache_path = settings.MONAI_LABEL_DATASTORE_CACHE_PATH
        self.cache_path = (
            os.path.join(cache_path, "sam2")
            if cache_path
            else os.path.join(pathlib.Path.home(), "cache", "monailabel", "sam2")
        )
        print('CACHE_CACHE_CACHE')
        print(self.cache_path)
        self.cached_dirs = {}
        self.cache_expiry_sec = 10 * 60

        remove_file(self.cache_path)
        os.makedirs(self.cache_path, exist_ok=True)
        logger.info(f"Image Cache Initialized: {self.cache_path}")

    def cleanup(self):
        ts = time()
        expired = {k: v for k, v in self.cached_dirs.items() if v < ts}
        for k, v in expired.items():
            self.cached_dirs.pop(k)
            logger.info(f"Remove Expired Image: {k}; ExpiryTs: {v}; CurrentTs: {ts}")
            remove_file(k)

    def monitor(self):
        self.cleanup()
        time_loop = Timeloop()
        schedule.every(1).minutes.do(self.cleanup)

        @time_loop.job(interval=timedelta(seconds=60))
        def run_scheduler():
            schedule.run_pending()

        time_loop.start(block=False)