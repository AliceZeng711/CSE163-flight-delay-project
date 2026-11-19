from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

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
FIGURE_DIR = PROJECT_ROOT / "figures"
SUMMARY_DIR = PROJECT_ROOT / "data" / "processed"
AIRPORT_ORDER = ["SEA", "DFW", "JFK"]


def find_data_file(possible_paths):
    """Return the first data path that exists."""
    for path in possible_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the required full-year data file."
    )


def load_merged_data():
    """Load and merge the full 2023 flight and weather datasets."""
    flight_path = find_data_file(FLIGHT_PATHS)
    weather_path = find_data_file(WEATHER_PATHS)

    flights = pd.read_csv(flight_path, low_memory=False)
    weather = pd.read_csv(weather_path, low_memory=False)
    merged = merge_flights_weather(flights, weather)

    matched = merged.dropna(subset=["weather_time_utc"]).copy()
    removed_count = len(merged) - len(matched)

    print(f"Flight rows before matching: {len(flights)}")
    print(f"Flight rows with weather: {len(matched)}")
    print(f"Flight rows without weather: {removed_count}")
    return matched


def precipitation_delay_summary(merged):
    """Summarize delay rates with and without measured precipitation."""
    trace = merged["precipitation_trace"].astype("string").str.lower()
    has_measurement = (
        merged["precipitation_mm"].notna()
        | trace.isin(["true", "false"])
    )
    precipitation = (
        merged["precipitation_mm"].fillna(0) > 0
    ) | trace.eq("true").fillna(False)

    analysis = merged.loc[has_measurement].copy()
    analysis["precipitation_status"] = precipitation.loc[
        has_measurement
    ].map({True: "Precipitation", False: "No precipitation"})

    return (
        analysis.groupby(
            ["origin", "precipitation_status"],
            as_index=False,
        )["delayed"]
        .agg(flight_count="count", delay_rate="mean")
    )


def visibility_delay_summary(merged):
    """Summarize delay rates for three visibility ranges."""
    analysis = merged.dropna(subset=["visibility_m"]).copy()
    labels = ["Low (<5 km)", "Medium (5-10 km)", "High (>10 km)"]
    analysis["visibility_group"] = pd.cut(
        analysis["visibility_m"],
        bins=[-float("inf"), 5000, 10000, float("inf")],
        labels=labels,
    )

    return (
        analysis.groupby(
            ["origin", "visibility_group"],
            observed=True,
            as_index=False,
        )["delayed"]
        .agg(flight_count="count", delay_rate="mean")
    )


def plot_precipitation_delay(summary):
    """Plot airport delay rates by precipitation status."""
    status_order = ["No precipitation", "Precipitation"]
    rates = summary.pivot(
        index="origin",
        columns="precipitation_status",
        values="delay_rate",
    ).reindex(index=AIRPORT_ORDER, columns=status_order)

    axis = rates.plot(
        kind="bar",
        figsize=(8, 5),
        color=["#4C78A8", "#F58518"],
    )
    axis.set_title("2023 Flight Delay Rates by Precipitation Status")
    axis.set_xlabel("Origin airport")
    axis.set_ylabel("Delayed flights")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Weather condition")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "precipitation_delay_rate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_visibility_delay(summary):
    """Plot airport delay rates for three visibility ranges."""
    visibility_order = [
        "Low (<5 km)",
        "Medium (5-10 km)",
        "High (>10 km)",
    ]
    rates = summary.pivot(
        index="origin",
        columns="visibility_group",
        values="delay_rate",
    ).reindex(index=AIRPORT_ORDER, columns=visibility_order)

    axis = rates.plot(
        kind="bar",
        figsize=(9, 5),
        color=["#E45756", "#F2CF5B", "#54A24B"],
    )
    axis.set_title("2023 Flight Delay Rates by Visibility")
    axis.set_xlabel("Origin airport")
    axis.set_ylabel("Delayed flights")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Visibility range")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "visibility_delay_rate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    """Create merged-data summaries and EDA figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    merged = load_merged_data()
    precipitation = precipitation_delay_summary(merged)
    visibility = visibility_delay_summary(merged)

    precipitation.to_csv(
        SUMMARY_DIR / "precipitation_delay_summary.csv",
        index=False,
    )
    visibility.to_csv(
        SUMMARY_DIR / "visibility_delay_summary.csv",
        index=False,
    )
    plot_precipitation_delay(precipitation)
    plot_visibility_delay(visibility)

    print("Saved precipitation and visibility summaries.")
    print("Saved two merged-data figures in the figures folder.")


if __name__ == "__main__":
    main()
