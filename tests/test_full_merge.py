from pathlib import Path

import pandas as pd

from merge_data import merge_flights_weather


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "clean_flights_2023.csv",
    PROJECT_ROOT / "data" / "clean_flights_2023.csv",
]
WEATHER_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "weather_all_clean.csv",
    PROJECT_ROOT / "weather_outputs" / "weather_all_clean.csv",
]
EXPECTED_AIRPORTS = {"SEA", "DFW", "JFK"}


def find_data_file(possible_paths):
    """Return the first full-year data path that exists."""
    for path in possible_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the required full-year data file."
    )


def load_full_data():
    """Load the cleaned full-year flight and weather datasets."""
    flight_path = find_data_file(FLIGHT_PATHS)
    weather_path = find_data_file(WEATHER_PATHS)
    flights = pd.read_csv(flight_path, low_memory=False)
    weather = pd.read_csv(weather_path, low_memory=False)
    return flights, weather


def test_full_inputs(flights, weather):
    """Check the keys and airports required for the full merge."""
    assert not flights.empty, "The full flight dataset is empty."
    assert not weather.empty, "The full weather dataset is empty."
    assert flights["flight_id"].is_unique, "Flight IDs are not unique."
    assert set(flights["origin"].unique()) == EXPECTED_AIRPORTS
    assert set(weather["airport"].unique()) == EXPECTED_AIRPORTS

    flight_times = pd.to_datetime(
        flights["scheduled_departure_utc"],
        errors="coerce",
        utc=True,
    )
    weather_times = pd.to_datetime(
        weather["weather_time_utc"],
        errors="coerce",
        utc=True,
    )
    assert flight_times.notna().all(), "Some flight times are invalid."
    assert weather_times.notna().all(), "Some weather times are invalid."


def test_full_merge(flights, weather):
    """Check the full-year flight-weather merge result."""
    merged = merge_flights_weather(flights, weather)
    matched = merged["weather_time_utc"].notna()
    matched_rows = merged.loc[matched]

    assert len(merged) == len(flights), "The merge changed row count."
    assert merged["flight_id"].is_unique, "Merged IDs are not unique."
    assert matched.mean() > 0.999, "Less than 99.9% of flights matched."
    assert (
        matched_rows["origin"] == matched_rows["airport"]
    ).all(), "A flight matched the wrong airport."
    assert (
        matched_rows["weather_time_utc"]
        <= matched_rows["weather_cutoff_utc"]
    ).all(), "A flight used weather that was not yet available."
    assert matched_rows["weather_age_hours"].between(
        0,
        3,
    ).all(), "A weather match is outside the three-hour window."

    unmatched_times = merged.loc[
        ~matched,
        "scheduled_departure_utc",
    ]
    year_boundary = pd.Timestamp("2024-01-01", tz="UTC")
    assert (
        unmatched_times >= year_boundary
    ).all(), "An unmatched flight is not from the UTC year boundary."
    return merged


def main():
    """Run all full-year merge tests."""
    flights, weather = load_full_data()
    test_full_inputs(flights, weather)
    merged = test_full_merge(flights, weather)

    matched_count = merged["weather_time_utc"].notna().sum()
    unmatched_count = len(merged) - matched_count
    print(f"Full-year flights: {len(merged)}")
    print(f"Matched flights: {matched_count}")
    print(f"Unmatched year-boundary flights: {unmatched_count}")
    print("All full-year merge tests passed!")


if __name__ == "__main__":
    main()
