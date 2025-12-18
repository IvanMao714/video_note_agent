import logging

TRACE = 15  # between DEBUG(10) and INFO(20)
logging.addLevelName(TRACE, "TRACE")

def trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)

logging.Logger.trace = trace
