import logging
import logging.config


log_dict = {
	'version': 1,
	'disable_existing_loggers': False,
	'formatters': {
		'standard': {
			'format': '%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s'
		},
	},
	'handlers': {
		'default': {
			'level': 'INFO',
			'formatter': 'standard',
			'class': 'logging.StreamHandler',
		},
		'file_handler': {
			'level': 'INFO',
			'filename': 'test.log',
			'class': 'logging.handlers.RotatingFileHandler',
			'formatter': 'standard',
			'mode' : 'a',
			'maxBytes': 10000000,
			'backupCount' : 100
		}
	},
	'loggers': {
		'': {
			'handlers': ['file_handler','default'],
			'level': 'INFO',
			'propagate': True
		},
	}
}

# --------------------------------
# set log file name and path
# ---------------------------------

global _logger
def init_log(filename):
	LOG_FILENAME = filename
	log_dict['handlers']['file_handler']['filename'] = LOG_FILENAME
	logging.config.dictConfig(log_dict)
	_logger = logging.getLogger(__name__)

	DECISION_LEVEL_NUM = 35
	logging.addLevelName(DECISION_LEVEL_NUM, "DECISION")

	def decision(self, message, *args, **kws):
		#Yes, logger takes its '*args' as 'args'.
		if self.isEnabledFor(DECISION_LEVEL_NUM):
			self._log(DECISION_LEVEL_NUM, message, args, **kws)
	logging.Logger.decision = decision
	return _logger

# _logger.decision("Processing %s source document", 'current_source_document')

# _logger.error('Error in case : ' + "current_source_document")


# handlers = _logger.handlers[:]
# for handler in handlers:
    # handler.close()
    # _logger.removeHandler(handler)
# del _logger