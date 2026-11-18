"""Clean 2023 NOAA weather data and create summary tables and plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Conservative policy: accept measurements with these quality flags.
ACCEPTED_QUALITY = {"0", "1", "4", "5"}

AIRPORT_FILES = [
    ("SEA", "72793024233", "sea.weather.2023.csv"),
    ("DFW", "72259003927", "dfw,weather.csv"),
    ("JFK", "74486094789", "jfk.weather.csv"),
]

WEATHER_COLUMNS = [
    "temperature_c",
    "wind_speed_mps",
    "visibility_m",
    "sea_level_pressure_hpa",
    "precipitation_mm",
    "precipitation_period_hours",
]

AIRPORT_COLORS = {
    "SEA": "#1f77b4",
    "DFW": "#ff7f0e",
    "JFK": "#2ca02c",
}


def split_field(data, column, width):
    """Return separated NOAA components from data[column].

    Missing columns and missing components remain missing.
    width specifies the expected number of components.
    """
    if column in data.columns:
        values = data[column].astype("string")
    else:
        values = pd.Series(
            pd.NA, index=data.index, dtype="string"
        )

    parts = values.str.split(",", expand=True)
    return parts.reindex(columns=range(width))


def clean_number(parts, value_index, quality_index, missing, scale):
    """Return decoded measurements from the component DataFrame parts.

    value_index and quality_index select the value and quality columns.
    Replace the missing code and rejected quality values with NaN,
    then divide valid measurements by scale.
    """
    values = pd.to_numeric(
        parts[value_index], errors="coerce"
    )
    quality = parts[quality_index].astype("string")

    valid = (
        values.notna()
        & (values != missing)
        & quality.isin(ACCEPTED_QUALITY)
    )
    return values.where(valid) / scale


def remove_duplicates(weather):
    """Return one complete observation per station and UTC timestamp.

    Prefer the row with the most valid weather measurements. Count
    precipitation amount once, without giving its period extra weight.
    Break ties using FM-15, then FM-16, then other report types.
    Remaining ties retain the first input row.
    """
    data = weather.copy()

    measurements = WEATHER_COLUMNS[:-1]
    data["_valid_count"] = data[measurements].notna().sum(axis=1)
    data["_report_rank"] = (
        data["report_type"]
        .map({"FM-15": 0, "FM-16": 1})
        .fillna(2)
    )
    data["_input_order"] = range(len(data))

    data = data.sort_values(
        [
            "station",
            "weather_time_utc",
            "_valid_count",
            "_report_rank",
            "_input_order",
        ],
        ascending=[True, True, False, True, True],
    )
    data = data.drop_duplicates(
        subset=["station", "weather_time_utc"],
        keep="first",
    )
    data = data.drop(
        columns=["_valid_count", "_report_rank", "_input_order"]
    )

    return data.sort_values(
        ["station", "weather_time_utc"]
    ).reset_index(drop=True)


def clean_weather(raw, airport, station):
    """Return cleaned observations for raw data, airport, and station.

    Preserve actual UTC observation times and missing measurements.
    Keep quality flags and precipitation conditions for review.
    Do not modify the input DataFrame.
    """
    required = {"STATION", "DATE", "TMP", "WND", "VIS", "SLP"}
    missing_columns = required - set(raw.columns)

    if missing_columns:
        raise ValueError(
            f"{airport}: missing columns {sorted(missing_columns)}"
        )

    stations = raw["STATION"].astype("string").str.strip()
    if not stations.eq(station).all():
        raise ValueError(
            f"{airport}: expected station {station}"
        )

    clean = pd.DataFrame(index=raw.index)
    clean["airport"] = airport
    clean["station"] = stations
    clean["weather_time_utc"] = pd.to_datetime(
        raw["DATE"], utc=True, errors="coerce"
    )

    times = clean["weather_time_utc"]
    wrong_year = times.notna() & (times.dt.year != 2023)
    if wrong_year.any():
        raise ValueError(
            f"{airport}: file contains observations outside 2023"
        )

    temperature = split_field(raw, "TMP", 2)
    wind = split_field(raw, "WND", 5)
    visibility = split_field(raw, "VIS", 4)
    pressure = split_field(raw, "SLP", 2)
    rain = split_field(raw, "AA1", 4)

    # Temperature is stored in tenths of degrees Celsius.
    clean["temperature_c"] = clean_number(
        temperature, 0, 1, 9999, 10
    )

    # WND components 4 and 5 contain speed and speed quality.
    clean["wind_speed_mps"] = clean_number(
        wind, 3, 4, 9999, 10
    )

    # Visibility is already measured in meters.
    clean["visibility_m"] = clean_number(
        visibility, 0, 1, 999999, 1
    )

    # Sea-level pressure is stored in tenths of hPa.
    clean["sea_level_pressure_hpa"] = clean_number(
        pressure, 0, 1, 99999, 10
    )

    # AA1 contains period, amount, condition, and quality.
    period = pd.to_numeric(rain[0], errors="coerce")
    condition = rain[2].astype("string")
    rain_quality = rain[3].astype("string")

    bad_condition = condition.isin(
        ["1", "3", "5", "6", "7"]
    )
    clean["precipitation_mm"] = clean_number(
        rain, 1, 3, 9999, 10
    ).mask(bad_condition)

    valid_period = (
        period.between(0, 98)
        & rain_quality.isin(ACCEPTED_QUALITY)
        & ~bad_condition
    )
    clean["precipitation_period_hours"] = period.where(
        valid_period
    )

    # Keep trace information; missing AA1 is never filled with zero.
    clean["precipitation_condition_code"] = condition
    clean["precipitation_trace"] = (
        condition.eq("2").astype("boolean")
    )

    for column in [
        "wind_speed_mps",
        "visibility_m",
        "precipitation_mm",
    ]:
        clean[column] = clean[column].where(
            clean[column] >= 0
        )

    valid_pressure = clean["sea_level_pressure_hpa"] > 0
    clean["sea_level_pressure_hpa"] = (
        clean["sea_level_pressure_hpa"].where(valid_pressure)
    )

    clean["temperature_quality"] = temperature[1]
    clean["wind_speed_quality"] = wind[4]
    clean["visibility_quality"] = visibility[1]
    clean["pressure_quality"] = pressure[1]
    clean["precipitation_quality"] = rain[3]

    if "REPORT_TYPE" in raw.columns:
        clean["report_type"] = (
            raw["REPORT_TYPE"].astype("string").str.strip()
        )
    else:
        clean["report_type"] = ""

    clean = clean.dropna(subset=["weather_time_utc"])
    return remove_duplicates(clean)


def summarize_weather(weather, output_folder):
    """Save per-airport numeric and missingness summaries.

    weather contains cleaned observations. output_folder is a Path.
    Return a table of missing counts and percentages by airport.
    """
    missing_tables = []
    numeric_tables = []

    for airport, group in weather.groupby("airport"):
        missing = pd.DataFrame({
            "missing_count": group[WEATHER_COLUMNS].isna().sum(),
            "missing_percent": (
                group[WEATHER_COLUMNS].isna().mean() * 100
            ),
        })
        missing.index.name = "variable"
        missing = missing.reset_index()
        missing["airport"] = airport
        missing_tables.append(missing)

        numeric = group[WEATHER_COLUMNS].describe().T
        numeric = numeric[
            ["mean", "std", "min", "25%", "50%", "75%", "max"]
        ]
        numeric.index.name = "variable"
        numeric = numeric.reset_index()
        numeric["airport"] = airport
        numeric_tables.append(numeric)

    missing = pd.concat(missing_tables, ignore_index=True)
    numeric = pd.concat(numeric_tables, ignore_index=True)

    missing.to_csv(
        output_folder / "weather_missing_summary.csv",
        index=False,
    )
    numeric.to_csv(
        output_folder / "weather_numeric_summary.csv",
        index=False,
    )

    for column in ["airport", "station", "report_type"]:
        counts = weather[column].value_counts(dropna=False)
        counts.to_csv(
            output_folder / f"counts_{column}.csv"
        )

    return missing


def plot_monthly_temperature(weather, output_folder):
    """Save monthly temperature means, coverage counts, and a plot.

    Average reports within each UTC hour first. Then average available
    hourly means within each month, so extra reports in a single hour
    do not give that hour more weight. Missing hours are excluded.
    """
    data = weather.copy()
    data["hour"] = data["weather_time_utc"].dt.floor("h")

    hourly = data.groupby(
        ["airport", "hour"]
    )["temperature_c"].mean()
    hourly = hourly.reset_index()
    hourly["month"] = hourly["hour"].dt.month

    monthly = hourly.groupby(
        ["airport", "month"]
    )["temperature_c"].agg(["mean", "count"])
    monthly = monthly.reset_index()

    monthly.to_csv(
        output_folder / "monthly_temperature.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for airport in ["SEA", "DFW", "JFK"]:
        airport_data = monthly[
            monthly["airport"] == airport
        ]
        ax.plot(
            airport_data["month"],
            airport_data["mean"],
            marker="o",
            label=airport,
            color=AIRPORT_COLORS[airport],
        )

    ax.set_title("Monthly Mean Temperature by Airport, 2023")
    ax.set_xlabel("Month (UTC)")
    ax.set_ylabel("Temperature (degrees C)")
    ax.set_xticks(range(1, 13))
    ax.legend(title="Airport")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_folder / "monthly_temperature.png",
        dpi=200,
    )
    plt.close(fig)


def plot_missingness(missing, output_folder):
    """Save a grouped bar plot of cleaned measurement missingness."""
    labels = {
        "temperature_c": "Temperature",
        "wind_speed_mps": "Wind speed",
        "visibility_m": "Visibility",
        "sea_level_pressure_hpa": "Pressure",
        "precipitation_mm": "Precipitation",
        "precipitation_period_hours": "Precipitation period",
    }

    data = missing.copy()
    data["variable"] = data["variable"].replace(labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=data,
        x="variable",
        y="missing_percent",
        hue="airport",
        hue_order=["SEA", "DFW", "JFK"],
        palette=AIRPORT_COLORS,
        ax=ax,
    )

    ax.set_title(
        "Missing Weather Measurements After Cleaning, 2023"
    )
    ax.set_xlabel("Weather variable")
    ax.set_ylabel("Missing observations (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Airport")

    fig.tight_layout()
    fig.savefig(
        output_folder / "weather_missingness.png",
        dpi=200,
    )
    plt.close(fig)


def save_sample(weather, output_folder):
    """Save SEA observations within the requested inclusive UTC range."""
    start = pd.Timestamp(
        "2023-01-10 06:00:00", tz="UTC"
    )
    end = pd.Timestamp(
        "2023-01-12 12:00:00", tz="UTC"
    )

    is_sea = weather["airport"] == "SEA"
    in_range = weather["weather_time_utc"].between(
        start, end
    )
    sample = weather[is_sea & in_range]

    if sample.empty:
        raise ValueError("No observations in the SEA sample range")

    sample.to_csv(
        output_folder / "weather_clean_sample.csv",
        index=False,
    )
    print("SEA sample rows:", len(sample))


def main():
    """Read three airport files and save 2023 weather EDA results."""
    folder = Path(__file__).resolve().parent
    output_folder = folder / "weather_outputs"
    output_folder.mkdir(exist_ok=True)

    clean_tables = []
    coverage_rows = []

    for airport, station, filename in AIRPORT_FILES:
        input_path = folder / filename
        raw = pd.read_csv(
            input_path,
            dtype={"STATION": "string"},
            low_memory=False,
        )
        clean = clean_weather(raw, airport, station)

        clean.to_csv(
            output_folder / f"weather_{airport}_2023_clean.csv",
            index=False,
        )
        clean_tables.append(clean)

        raw_times = pd.to_datetime(
            raw["DATE"], utc=True, errors="coerce"
        )
        invalid_times = int(raw_times.isna().sum())
        duplicates_removed = (
            len(raw) - invalid_times - len(clean)
        )

        coverage_rows.append({
            "airport": airport,
            "station": station,
            "raw_rows": len(raw),
            "raw_columns": len(raw.columns),
            "clean_rows": len(clean),
            "invalid_times_removed": invalid_times,
            "duplicate_rows_removed": duplicates_removed,
            "first_time_utc": clean["weather_time_utc"].min(),
            "last_time_utc": clean["weather_time_utc"].max(),
            "days_observed": (
                clean["weather_time_utc"].dt.date.nunique()
            ),
        })

        print(
            f"{airport}: {len(raw)} raw rows -> "
            f"{len(clean)} clean rows"
        )

    weather = pd.concat(clean_tables, ignore_index=True)
    weather = weather.sort_values(
        ["airport", "weather_time_utc"]
    ).reset_index(drop=True)

    weather.to_csv(
        output_folder / "weather_all_clean.csv",
        index=False,
    )

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        output_folder / "weather_coverage.csv",
        index=False,
    )

    save_sample(weather, output_folder)
    missing = summarize_weather(weather, output_folder)
    plot_monthly_temperature(weather, output_folder)
    plot_missingness(missing, output_folder)

    print("Results saved to:", output_folder)


if __name__ == "__main__":
    main()
