from enum import Enum

class ResultType(Enum):
    P_VALUES = ("p_values", ".csv")
    SUMMARY_STATS = ("summary_stats", ".yaml")
    MAX_TEST_STATS = ("max_test_stats", ".csv")

    def __init__(self, stem: str, suffix: str):
        self.stem = stem  # base name of file
        self.suffix = suffix  # file extension