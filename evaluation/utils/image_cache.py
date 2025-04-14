# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file contains code from the SAM2 integration into MONAI Label:
# https://github.com/Project-MONAI/MONAILabel/blob/main/monailabel/sam2/infer.py

import logging
import os
import pathlib
from datetime import timedelta
from time import time

import schedule
from timeloop import Timeloop
from monailabel.utils.others.generic import remove_file

logger = logging.getLogger("evaluation_pipeline_logger")


class ImageCache:
    def __init__(self, cache_path):
        self.cache_path = (
            os.path.join(cache_path, "sam2")
            if cache_path
            else os.path.join(pathlib.Path.home(), ".cache", "monailabel", "sam2")
        )
        self.cached_dirs = {}
        self.cache_expiry_sec = 20 * 60

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
