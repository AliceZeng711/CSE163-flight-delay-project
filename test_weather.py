"""Test weather cleaning with small, predictable NOAA datasets."""

from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from weather_analysis import clean_weather, save_sample


def make_raw_data():
    """Return a small NOAA dataset with known values and edge cases."""
    return pd.DataFrame({
        "STATION": ["72793024233"] * 5,
        "DATE": [
            "2023-01-10T06:53:00",
            "2023-01-10T06:53:00",
            "2023-01-10T07:53:00",
            "2023-01-10T08:53:00",
            "invalid",
        ],
        "TMP": [
            "+9999,9",
            "+0072,1",
            "+0080,3",
            "-0015,5",
            "+0072,1",
        ],
        "WND": [
            "250,1,N,0051,1",
            "250,1,N,0051,1",
            "999,9,9,9999,9",
            "180,1,N,0000,1",
            "250,1,N,0051,1",
        ],
        "VIS": [
            "016093,1,N,1",
            "016093,1,N,1",
            "999999,9,9,9",
            "008000,5,N,5",
            "016093,1,N,1",
        ],
        "SLP": [
            "10132,1",
            "10132,1",
            "99999,9",
            "10000,5",
            "10132,1",
        ],
        "AA1": [
            None,
            "01,0010,9,1",
            None,
            "06,0000,2,1",
            None,
        ],
        "REPORT_TYPE": [
            "FM-15",
            "FM-15",
            "FM-15",
            "FM-15",
            "FM-15",
        ],
    })


def test_unit_conversions():
    """Check temperature, wind, visibility, pressure, and rainfall units."""
    clean = clean_weather(
        make_raw_data(), "SEA", "72793024233"
    )
    first = clean.iloc[0]

    assert abs(first["temperature_c"] - 7.2) < 0.000001
    assert abs(first["wind_speed_mps"] - 5.1) < 0.000001
    assert first["visibility_m"] == 16093
    assert abs(
        first["sea_level_pressure_hpa"] - 1013.2
    ) < 0.000001
    assert first["precipitation_mm"] == 1.0
    assert first["precipitation_period_hours"] == 1

    last = clean.iloc[-1]
    assert last["temperature_c"] == -1.5
    assert last["wind_speed_mps"] == 0
    assert last["visibility_m"] == 8000
    assert last["sea_level_pressure_hpa"] == 1000


def test_missing_values_and_quality():
    """Check missing codes and rejected quality flags remain missing."""
    clean = clean_weather(
        make_raw_data(), "SEA", "72793024233"
    )
    missing_row = clean.iloc[1]

    # Temperature has a plausible value but a rejected quality flag.
    assert pd.isna(missing_row["temperature_c"])
    assert pd.isna(missing_row["wind_speed_mps"])
    assert pd.isna(missing_row["visibility_m"])
    assert pd.isna(missing_row["sea_level_pressure_hpa"])

    # Missing precipitation must not become zero rainfall.
    assert pd.isna(missing_row["precipitation_mm"])
    assert pd.isna(missing_row["precipitation_period_hours"])

    trace_row = clean.iloc[-1]
    assert trace_row["precipitation_trace"]
    assert trace_row["precipitation_period_hours"] == 6


def test_duplicates_and_utc():
    """Check duplicate selection, invalid dates, and exact UTC minutes."""
    clean = clean_weather(
        make_raw_data(), "SEA", "72793024233"
    )

    # Five input rows minus one duplicate and one invalid timestamp.
    assert len(clean) == 3
    assert not clean.duplicated(
        ["station", "weather_time_utc"]
    ).any()
    assert clean["weather_time_utc"].notna().all()

    first = clean.iloc[0]
    assert first["weather_time_utc"] == pd.Timestamp(
        "2023-01-10T06:53:00Z"
    )
    assert first["weather_time_utc"].minute == 53

    # The more complete duplicate should be retained.
    assert first["temperature_c"] == 7.2
    assert first["precipitation_mm"] == 1.0
    assert clean["airport"].eq("SEA").all()
    assert clean["station"].eq("72793024233").all()


def test_input_unchanged():
    """Check that cleaning does not modify the original DataFrame."""
    raw = make_raw_data()
    original = raw.copy(deep=True)

    clean_weather(raw, "SEA", "72793024233")

    pd.testing.assert_frame_equal(raw, original)


def test_wrong_station():
    """Check that an incorrect airport station raises ValueError."""
    raw = make_raw_data()
    raw["STATION"] = "72259003927"

    try:
        clean_weather(raw, "SEA", "72793024233")
    except ValueError:
        return

    raise AssertionError("An incorrect station should raise ValueError")


def test_sample_time_range():
    """Check inclusive sample boundaries and exclusion of other airports."""
    weather = pd.DataFrame({
        "airport": ["SEA", "SEA", "SEA", "SEA", "SEA", "DFW"],
        "weather_time_utc": pd.to_datetime([
            "2023-01-10T05:59:00Z",
            "2023-01-10T06:00:00Z",
            "2023-01-11T08:53:00Z",
            "2023-01-12T12:00:00Z",
            "2023-01-12T12:01:00Z",
            "2023-01-11T08:53:00Z",
        ], utc=True),
    })

    with TemporaryDirectory() as directory:
        output_folder = Path(directory)
        save_sample(weather, output_folder)

        saved = pd.read_csv(
            output_folder / "weather_clean_sample.csv"
        )
        times = pd.to_datetime(
            saved["weather_time_utc"], utc=True
        )

        assert len(saved) == 3
        assert saved["airport"].eq("SEA").all()
        assert times.min() == pd.Timestamp(
            "2023-01-10T06:00:00Z"
        )
        assert times.max() == pd.Timestamp(
            "2023-01-12T12:00:00Z"
        )


def main():
    """Run all weather tests and report success."""
    test_unit_conversions()
    test_missing_values_and_quality()
    test_duplicates_and_utc()
    test_input_unchanged()
    test_wrong_station()
    test_sample_time_range()
    print("All weather tests passed!")


if __name__ == "__main__":
    main()
