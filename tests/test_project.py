"""Tests for the cleaned weather sample used in the CSE 163 project."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEATHER_SAMPLE = (
    PROJECT_ROOT / "data" / "raw" / "weather_clean_sample.csv"
)

REQUIRED_COLUMNS = {
    "airport",
    "station",
    "weather_time_utc",
    "temperature_c",
    "wind_speed_mps",
    "visibility_m",
    "sea_level_pressure_hpa",
    "precipitation_mm",
    "precipitation_period_hours",
    "precipitation_condition_code",
    "precipitation_trace",
    "temperature_quality",
    "wind_speed_quality",
    "visibility_quality",
    "pressure_quality",
    "precipitation_quality",
    "report_type",
}

NUMERIC_COLUMNS = [
    "temperature_c",
    "wind_speed_mps",
    "visibility_m",
    "sea_level_pressure_hpa",
    "precipitation_mm",
    "precipitation_period_hours",
    "precipitation_condition_code",
]


def load_weather_sample(path=WEATHER_SAMPLE):
    """Load Wendy's cleaned weather sample."""
    assert path.exists(), f"Weather sample was not found: {path}"
    return pd.read_csv(path)


def test_required_columns(weather):
    """Check that the sample is nonempty and has the expected columns."""
    missing_columns = REQUIRED_COLUMNS - set(weather.columns)
    assert len(weather) > 0, "The weather sample is empty."
    assert not missing_columns, (
        f"Missing required weather columns: {sorted(missing_columns)}"
    )


def test_airport_and_station(weather):
    """Check the airport and weather-station identifiers."""
    valid_airports = {"SEA", "DFW", "JFK"}
    airports = set(weather["airport"].dropna())

    assert weather["airport"].notna().all(), "Some airports are missing."
    assert weather["station"].notna().all(), "Some stations are missing."
    assert airports <= valid_airports, (
        f"Unexpected airport codes: {sorted(airports - valid_airports)}"
    )


def test_weather_times(weather):
    """Check that UTC times parse, are sorted, and are unique per airport."""
    weather_times = pd.to_datetime(
        weather["weather_time_utc"], errors="coerce", utc=True
    )
    keys = weather.assign(parsed_weather_time=weather_times)

    assert weather_times.notna().all(), "Some weather times are invalid."
    for airport, airport_rows in keys.groupby("airport"):
        assert airport_rows["parsed_weather_time"].is_monotonic_increasing, (
            f"Weather observations for {airport} are not sorted by time."
        )
    assert not keys.duplicated(
        subset=["airport", "parsed_weather_time"]
    ).any(), "Duplicate airport-time weather records were found."


def test_numeric_columns(weather):
    """Check that each nonmissing numeric value can be parsed as a number."""
    for column in NUMERIC_COLUMNS:
        original_values = weather[column]
        numeric_values = pd.to_numeric(original_values, errors="coerce")
        originally_present = original_values.notna()

        assert numeric_values[originally_present].notna().all(), (
            f"Column {column} contains a nonnumeric value."
        )


def test_weather_ranges(weather):
    """Check broad physical ranges without changing missing values."""
    temperature = pd.to_numeric(
        weather["temperature_c"], errors="coerce"
    ).dropna()
    wind_speed = pd.to_numeric(
        weather["wind_speed_mps"], errors="coerce"
    ).dropna()
    visibility = pd.to_numeric(
        weather["visibility_m"], errors="coerce"
    ).dropna()
    pressure = pd.to_numeric(
        weather["sea_level_pressure_hpa"], errors="coerce"
    ).dropna()
    precipitation = pd.to_numeric(
        weather["precipitation_mm"], errors="coerce"
    ).dropna()
    precipitation_period = pd.to_numeric(
        weather["precipitation_period_hours"], errors="coerce"
    ).dropna()

    assert temperature.between(-90, 60).all(), (
        "A temperature is outside the expected range."
    )
    assert (wind_speed >= 0).all(), "A wind speed is negative."
    assert (visibility >= 0).all(), "A visibility value is negative."
    assert pressure.between(800, 1100).all(), (
        "A pressure value is outside the expected range."
    )
    assert (precipitation >= 0).all(), "A precipitation value is negative."
    assert (precipitation_period > 0).all(), (
        "A precipitation period is not positive."
    )


def test_missing_value_policy(weather):
    """Keep merge keys complete while allowing missing weather readings."""
    core_weather = [
        "temperature_c",
        "wind_speed_mps",
        "visibility_m",
        "sea_level_pressure_hpa",
        "precipitation_mm",
    ]

    assert weather["airport"].notna().all()
    assert weather["weather_time_utc"].notna().all()
    assert weather["report_type"].notna().all()

    for column in core_weather:
        assert weather[column].notna().any(), (
            f"Column {column} has no usable values."
        )


def test_precipitation_fields(weather):
    """Check consistency among nonmissing precipitation fields."""
    precipitation_present = weather["precipitation_mm"].notna()
    trace_values = weather["precipitation_trace"].dropna().astype(str)

    assert weather.loc[
        precipitation_present, "precipitation_period_hours"
    ].notna().all(), (
        "A precipitation value is missing its measurement period."
    )
    assert trace_values.isin(["True", "False"]).all(), (
        "Precipitation trace contains a value other than True or False."
    )


def run_all_tests():
    """Run all tests for the cleaned weather sample."""
    weather = load_weather_sample()
    test_required_columns(weather)
    test_airport_and_station(weather)
    test_weather_times(weather)
    test_numeric_columns(weather)
    test_weather_ranges(weather)
    test_missing_value_policy(weather)
    test_precipitation_fields(weather)


if __name__ == "__main__":
    run_all_tests()
    print("All weather sample tests passed!")
