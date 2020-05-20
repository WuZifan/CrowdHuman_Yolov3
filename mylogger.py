# -*- coding: UTF-8 -*-
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
# import config as cfg


class MyLogger:
    def __init__(self, filename='FENCE_LOG', roll_time='H', interval=1,logger_name =''):
        """
        :param filename:
        :param roll_time:   “S”: Seconds
                            “M”: Minutes
                            “H”: Hours
                            “D”: Days
                            “W”: Week day (0=Monday)
                            “midnight”: Roll over at midnight
        :param interval:    一周期内滚动频次，整数，1，2，3，4
        :param logger_name: 一定不能少，用于区分不同的logger
        """
        if logger_name =='':
            self.__log = logging.getLogger('default')
        else:
            self.__log = logging.getLogger(logger_name)
        self.__log.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s")

        log_file_handler = TimedRotatingFileHandler(filename=filename, when=roll_time, interval=interval)
        log_file_handler.setFormatter(formatter)
        log_file_handler.setLevel(logging.DEBUG)
        self.__log.addHandler(log_file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.__log.addHandler(stream_handler)

        self.info = self.__log.info
        self.error = self.__log.error
        self.debug = self.__log.debug
        self.warning = self.__log.warning

# my_logger = MyLogger(cfg.LOG_PATH+'MAIN_PROCESS.log')
# print('hahahah')

