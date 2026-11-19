"""Backup of the unsaved merge tests recovered from VS Code."""

import pandas as pd

from merge_data import merge_flights_weather


def make_test_flights():
    """Create small flight data with known expected matches."""
    return pd.DataFrame({
        "flight_id": [1, 2, 3, 4],
        "origin": ["SEA", "SEA", "DFW", "JFK"],
        "scheduled_departure_utc": pd.to_datetime([
            "2023-01-10 15:00:00+00:00",
            "2023-01-10 18:00:00+00:00",
            "2023-01-10 15:00:00+00:00",
            "2023-01-10 15:00:00+00:00",
        ]),
    })


def make_test_weather():
    """Create weather records before and after flight cutoffs."""
    return pd.DataFrame({
        "airport": ["SEA", "SEA", "SEA", "DFW"],
        "weather_time_utc": pd.to_datetime([
            "2023-01-10 12:50:00+00:00",
            "2023-01-10 13:10:00+00:00",
            "2023-01-10 15:40:00+00:00",
            "2023-01-10 12:40:00+00:00",
        ]),
        "temperature_c": [8.0, 9.0, 10.0, 20.0],
        "wind_speed_mps": [2.0, 2.5, 3.0, 4.0],
        "visibility_m": [14000, 15000, 16000, 12000],
    })


def test_row_count_and_unique_flights():
    flights = make_test_flights()
    weather = make_test_weather()
    result = merge_flights_weather(flights, weather)

    assert len(result) == len(flights)
    assert result["flight_id"].is_unique


def test_expected_weather_matches():
    flights = make_test_flights()
    weather = make_test_weather()
    result = merge_flights_weather(flights, weather)
    result = result.set_index("flight_id")

    assert result.loc[1, "weather_time_utc"] == pd.Timestamp(
        "2023-01-10 12:50:00+00:00"
    )
    assert result.loc[2, "weather_time_utc"] == pd.Timestamp(
        "2023-01-10 15:40:00+00:00"
    )
    assert result.loc[3, "weather_time_utc"] == pd.Timestamp(
        "2023-01-10 12:40:00+00:00"
    )
    assert pd.isna(result.loc[4, "weather_time_utc"])


def test_no_future_weather():
    flights = make_test_flights()
    weather = make_test_weather()
    result = merge_flights_weather(flights, weather)
    matched = result.dropna(subset=["weather_time_utc"])

    assert (
        matched["weather_time_utc"] <= matched["weather_cutoff_utc"]
    ).all()


def test_airports_match():
    flights = make_test_flights()
    weather = make_test_weather()
    result = merge_flights_weather(flights, weather)
    matched = result.dropna(subset=["weather_time_utc"])

    assert (matched["origin"] == matched["airport"]).all()


def test_weather_age_limit():
    flights = pd.DataFrame({
        "flight_id": [10],
        "origin": ["SEA"],
        "scheduled_departure_utc": pd.to_datetime([
            "2023-01-10 15:00:00+00:00",
        ]),
    })
    weather = pd.DataFrame({
        "airport": ["SEA"],
        "weather_time_utc": pd.to_datetime([
            "2023-01-10 09:00:00+00:00",
        ]),
        "temperature_c": [8.0],
    })
    result = merge_flights_weather(
        flights,
        weather,
        cutoff_hours=2,
        max_weather_age_hours=3,
    )

    assert pd.isna(result.loc[0, "weather_time_utc"])


def main():
    test_row_count_and_unique_flights()
    test_expected_weather_matches()
    test_no_future_weather()
    test_airports_match()
    test_weather_age_limit()
    print("All tests passed!")


if __name__ == "__main__":
    main()
