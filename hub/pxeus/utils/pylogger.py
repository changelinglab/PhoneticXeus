import logging
from typing import Mapping, Optional


class RankedLogger(logging.LoggerAdapter):
    """Simplified logger for single-process inference (no Lightning)."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs
    ) -> None:
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)
