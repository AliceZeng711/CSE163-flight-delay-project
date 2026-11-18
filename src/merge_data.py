from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_SAMPLE = (
    PROJECT_ROOT / "data" / "raw" / "clean_flights_sea_2023_01.csv"
)
WEATHER_SAMPLE = (
    PROJECT_ROOT / "data" / "raw" / "weather_clean_sample.csv"
)
MERGED_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "merged_sample.csv"
)


def merge_flights_weather(
        flights,
        weather,
        cutoff_hours=2,
        max_weather_age_hours=3
):
    """Match each flight with the most recent usable weather record."""
    flights = flights.copy()
    weather = weather.copy()

    flights["scheduled_departure_utc"] = pd.to_datetime(
        flights["scheduled_departure_utc"],
        errors="coerce",
        utc=True,
    )
    weather["weather_time_utc"] = pd.to_datetime(
        weather["weather_time_utc"],
        errors="coerce",
        utc=True,
    )

    flights["weather_cutoff_utc"] = (
        flights["scheduled_departure_utc"]
        - pd.to_timedelta(cutoff_hours, unit="h")
    )

    flights["_original_order"] = range(len(flights))

    flights = flights.sort_values(
        ["weather_cutoff_utc", "origin"]
    )
    weather = weather.sort_values(
        ["weather_time_utc", "airport"]
    )

    merged = pd.merge_asof(
        flights,
        weather,
        left_on="weather_cutoff_utc",
        right_on="weather_time_utc",
        left_by="origin",
        right_by="airport",
        direction="backward",
        tolerance=pd.Timedelta(hours=max_weather_age_hours),
    )

    merged["weather_age_hours"] = (
        merged["weather_cutoff_utc"] - merged["weather_time_utc"]
    ).dt.total_seconds() / 3600

    merged = merged.sort_values("_original_order")
    merged = merged.drop(columns="_original_order")
    return merged.reset_index(drop=True)


def main():
    """Merge the cleaned flight and weather samples."""
    flights = pd.read_csv(FLIGHT_SAMPLE)
    weather = pd.read_csv(WEATHER_SAMPLE)

    merged = merge_flights_weather(flights, weather)
    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUTPUT, index=False)

    matched_count = merged["weather_time_utc"].notna().sum()

    print(f"Saved {len(merged)} flights to {MERGED_OUTPUT}")
    print(f"Flights matched with weather: {matched_count}")


if __name__ == "__main__":
    main()
