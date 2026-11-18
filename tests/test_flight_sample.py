"""Tests for Alice's cleaned flight sample."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_SAMPLE = (
    PROJECT_ROOT / "data" / "raw" / "clean_flights_sea_2023_01.csv"
)

REQUIRED_COLUMNS = {
    "flight_id",
    "flight_date",
    "day_of_week",
    "airline",
    "flight_number",
    "origin",
    "destination",
    "scheduled_departure_time",
    "scheduled_departure_local",
    "scheduled_departure_utc",
    "departure_delay",
    "delayed",
    "scheduled_duration",
    "distance",
}


def load_flight_sample(path=FLIGHT_SAMPLE):
    """Load Alice's cleaned flight sample."""
    assert path.exists(), f"Flight sample was not found: {path}"
    return pd.read_csv(path)


def test_required_columns(flights):
    """Check that the sample is nonempty and has the required columns."""
    missing_columns = REQUIRED_COLUMNS - set(flights.columns)

    assert len(flights) > 0, "The flight sample is empty."
    assert not missing_columns, (
        f"Missing required flight columns: {sorted(missing_columns)}"
    )


def test_no_missing_required_values(flights):
    """Check that every required column has complete values."""
    missing_counts = flights[list(REQUIRED_COLUMNS)].isna().sum()
    missing_columns = missing_counts[missing_counts > 0].to_dict()

    assert not missing_columns, (
        f"Required columns contain missing values: {missing_columns}"
    )


def test_unique_flight_ids(flights):
    """Check that every flight has a unique identifier."""
    duplicate_count = flights["flight_id"].duplicated().sum()

    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate flight IDs."
    )


def test_origin_airport(flights):
    """Check that this sample contains only SEA departures."""
    origins = set(flights["origin"].dropna())

    assert origins == {"SEA"}, (
        f"Expected only SEA as the origin, but found: {sorted(origins)}"
    )


def test_scheduled_departure_utc(flights):
    """Check that every scheduled UTC departure time can be parsed."""
    departure_times = pd.to_datetime(
        flights["scheduled_departure_utc"],
        errors="coerce",
        utc=True,
    )

    invalid_count = departure_times.isna().sum()

    assert invalid_count == 0, (
        f"Found {invalid_count} invalid scheduled UTC departure times."
    )


def test_delayed_label(flights):
    """Check that delayed means a departure delay of at least 15 minutes."""
    assert pd.api.types.is_bool_dtype(flights["delayed"]), (
        "The delayed column must contain Boolean values."
    )

    expected_delayed = flights["departure_delay"] >= 15
    mismatch_count = (
        flights["delayed"] != expected_delayed
    ).sum()

    assert mismatch_count == 0, (
        f"Found {mismatch_count} incorrect delayed labels."
    )


def run_all_tests():
    """Run all tests for Alice's cleaned flight sample."""
    flights = load_flight_sample()
    test_required_columns(flights)
    test_no_missing_required_values(flights)
    test_unique_flight_ids(flights)
    test_origin_airport(flights)
    test_scheduled_departure_utc(flights)
    test_delayed_label(flights)


if __name__ == "__main__":
    run_all_tests()
    print("All flight sample tests passed!")
