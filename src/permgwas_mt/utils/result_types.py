from enum import Enum

class ResultType(Enum):
    P_VALUES = ("p_values", ".csv")
    SUMMARY_STATS = ("summary_stats", ".txt")
    MIN_P_VALUES = ("min_p_values", ".csv")

    def __init__(self, stem: str, suffix: str):
        self.stem = stem  # base name of file
        self.suffix = suffix  # file extension