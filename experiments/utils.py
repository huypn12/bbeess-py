from datetime import datetime
import logging


class DurationLogger(object):
    def __init__(self):
        self.current_timepoint = datetime.now()

    def 