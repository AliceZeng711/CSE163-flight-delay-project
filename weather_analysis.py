"""Clean SEA NOAA weather data and export a sample for matching."""

from pathlib import Path

import pandas as pd


# Conservative policy: retain only these quality flags.
ACCEPTED_QUALITY = {"0", "1", "4", "5"}

WEATHER_COLUMNS = [
    "temperature_c",
    "wind_speed_mps",
    "visibility_m",
    "sea_level_pressure_hpa",
    "precipitation_mm",
    "precipitation_period_hours",
]


def split_field(data, column, width):
    """Split a NOAA field, allowing missing columns and components."""
    if column in data.columns:
        values = data[column].astype("string")
    else:
        values = pd.Series(pd.NA, index=data.index, dtype="string")

    return (
        values.str.split(",", expand=True)
        .reindex(columns=range(width))
    )


def clean_number(parts, value_index, quality_index, missing, scale):
    """Convert one encoded measurement using a conservative QC policy."""
    raw = pd.to_numeric(parts[value_index], errors="coerce")
    quality = parts[quality_index].astype("string")

    valid = (
        raw.notna()
        & raw.ne(missing)
        & quality.isin(ACCEPTED_QUALITY)
    )

    return raw.where(valid) / scale


def clean_weather(raw):
    """Decode SEA observations and keep one record per station-time.

    Duplicate rule:
    1. Prefer the row with the most valid weather measurements.
    2. If tied, prefer FM-15, then FM-16, then other report types.
    3. If still tied, retain the first row in the input file.

    Entire rows are selected; values from different reports are not mixed.
    """
    required = {"STATION", "DATE", "TMP", "WND", "VIS", "SLP"}
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = raw.copy()
    data["STATION"] = data["STATION"].astype("string").str.strip()

    # Prevent accidental use of a different airport's file.
    if not data["STATION"].eq("72793024233").all():
        raise ValueError("The input must contain only SEA station 72793024233.")

    clean = pd.DataFrame(index=data.index)
    clean["airport"] = "SEA"
    clean["station"] = data["STATION"]
    clean["weather_time_utc"] = pd.to_datetime(
        data["DATE"], utc=True, errors="coerce"
    )

    # TMP: temperature in tenths of degrees Celsius, then quality.
    temperature = split_field(data, "TMP", 2)
    clean["temperature_c"] = clean_number(
        temperature, 0, 1, 9999, 10
    )

    # WND: direction, direction quality, type, speed, speed quality.
    wind = split_field(data, "WND", 5)
    clean["wind_speed_mps"] = clean_number(
        wind, 3, 4, 9999, 10
    )

    # VIS: visibility in meters, quality, variability, variability quality.
    visibility = split_field(data, "VIS", 4)
    clean["visibility_m"] = clean_number(
        visibility, 0, 1, 999999, 1
    )

    # SLP: sea-level pressure in tenths of hPa, then quality.
    pressure = split_field(data, "SLP", 2)
    clean["sea_level_pressure_hpa"] = clean_number(
        pressure, 0, 1, 99999, 10
    )

    # AA1: accumulation hours, depth in tenths of mm, condition, quality.
    precipitation = split_field(data, "AA1", 4)
    period = pd.to_numeric(precipitation[0], errors="coerce")
    condition = precipitation[2].astype("string")
    quality = precipitation[3].astype("string")

    clean["precipitation_mm"] = clean_number(
        precipitation, 1, 3, 9999, 10
    )

    # Invalid/problem-period conditions do not represent usable amounts.
    bad_condition = condition.isin(["1", "3", "5", "6", "7"])
    clean["precipitation_mm"] = clean["precipitation_mm"].mask(
        bad_condition
    )

    valid_period = (
        period.between(0, 98)
        & quality.isin(ACCEPTED_QUALITY)
        & ~bad_condition
    )
    clean["precipitation_period_hours"] = period.where(valid_period)

    # Retain trace/condition information rather than silently losing it.
    clean["precipitation_condition_code"] = condition
    clean["precipitation_trace"] = condition.eq("2").astype("boolean")

    # Reject impossible negative values for nonnegative measurements.
    for column in [
        "wind_speed_mps",
        "visibility_m",
        "precipitation_mm",
    ]:
        clean[column] = clean[column].where(clean[column].ge(0))

    clean["sea_level_pressure_hpa"] = (
        clean["sea_level_pressure_hpa"]
        .where(clean["sea_level_pressure_hpa"].gt(0))
    )

    # Preserve quality flags to make cleaning decisions inspectable.
    clean["temperature_quality"] = temperature[1]
    clean["wind_speed_quality"] = wind[4]
    clean["visibility_quality"] = visibility[1]
    clean["pressure_quality"] = pressure[1]
    clean["precipitation_quality"] = precipitation[3]

    if "REPORT_TYPE" in data:
        clean["report_type"] = (
            data["REPORT_TYPE"].astype("string").str.strip()
        )
    else:
        clean["report_type"] = ""

    invalid_times = clean["weather_time_utc"].isna().sum()
    clean = clean.dropna(subset=["weather_time_utc"]).copy()

    clean["_valid_count"] = clean[WEATHER_COLUMNS].notna().sum(axis=1)
    clean["_report_rank"] = (
        clean["report_type"]
        .map({"FM-15": 0, "FM-16": 1})
        .fillna(2)
    )
    clean["_input_order"] = range(len(clean))

    before_deduplication = len(clean)
    clean = clean.sort_values(
        [
            "station",
            "weather_time_utc",
            "_valid_count",
            "_report_rank",
            "_input_order",
        ],
        ascending=[True, True, False, True, True],
    )
    clean = clean.drop_duplicates(
        subset=["station", "weather_time_utc"],
        keep="first",
    )
    removed_duplicates = before_deduplication - len(clean)

    clean = (
        clean.drop(
            columns=["_valid_count", "_report_rank", "_input_order"]
        )
        .sort_values(["station", "weather_time_utc"])
        .reset_index(drop=True)
    )

    print("Raw rows:", len(raw))
    print("Invalid timestamps removed:", invalid_times)
    print("Duplicate station-time rows removed:", removed_duplicates)
    print("Clean rows:", len(clean))

    return clean


def main():
    """Clean the full SEA file and export the requested UTC sample."""
    folder = Path(__file__).resolve().parent
    input_path = folder / "sea.weather.2023.csv"

    raw = pd.read_csv(
        input_path,
        dtype={"STATION": "string"},
        low_memory=False,
    )
    clean = clean_weather(raw)

    start = pd.Timestamp("2023-01-10 06:00:00", tz="UTC")
    end = pd.Timestamp("2023-01-12 12:00:00", tz="UTC")

    sample = clean.loc[
        clean["weather_time_utc"].between(start, end)
    ].copy()

    if sample.empty:
        raise ValueError("No observations found in the requested range.")

    # Basic checks; additional unit tests should be written separately.
    assert not clean.duplicated(
        ["station", "weather_time_utc"]
    ).any()
    assert sample["weather_time_utc"].between(start, end).all()
    assert sample["airport"].eq("SEA").all()

    missing_summary = pd.DataFrame({
        "missing_count": clean[WEATHER_COLUMNS].isna().sum(),
        "missing_percent": (
            clean[WEATHER_COLUMNS].isna().mean() * 100
        ),
    })
    missing_summary.index.name = "variable"

    numeric_summary = (
        clean[WEATHER_COLUMNS]
        .describe()
        .loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    )

    # Missing measurements remain blank in CSV, never filled with zero.
    clean.to_csv(folder / "weather_SEA_2023_clean.csv", index=False)
    sample.to_csv(folder / "weather_clean_sample.csv", index=False)
    missing_summary.to_csv(folder / "weather_missing_summary.csv")
    numeric_summary.to_csv(folder / "weather_numeric_summary.csv")

    print("\nSample rows:", len(sample))
    print("First observation:", sample["weather_time_utc"].min())
    print("Last observation:", sample["weather_time_utc"].max())
    print("\nMissing values after cleaning:")
    print(missing_summary.round(2))
    print("\nSample preview:")
    print(sample[[
        "weather_time_utc",
        "temperature_c",
        "wind_speed_mps",
        "precipitation_mm",
    ]].head())
    print("\nSaved files to:", folder)


if __name__ == "__main__":
    main()