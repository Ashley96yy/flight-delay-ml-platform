import pytest

from flight_delay.features.schema import (
    BASELINE_FEATURES,
    TARGETS,
    validate_baseline_features,
)


def test_targets_not_in_features():
    """Target labels must never be baseline predictors."""
    for target in TARGETS.values():
        assert target not in BASELINE_FEATURES


def test_valid_features():
    """Approved baseline features should pass validation."""
    columns = ["FlightDate", "Origin", "Dest", "Distance"]

    validate_baseline_features(columns)


@pytest.mark.parametrize(
    "leaked_column",
    [
        "DepDelay",
        "ArrDelay",
        "DepTime",
        "ArrTime",
        "DepDel15",
        "ArrDel15",
    ],
)
def test_leakage_features_rejected(leaked_column):
    """Current-flight outcomes must be rejected."""
    columns = ["Origin", "Dest", leaked_column]

    with pytest.raises(ValueError):
        validate_baseline_features(columns)
