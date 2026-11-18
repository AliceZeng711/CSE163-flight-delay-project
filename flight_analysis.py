import pandas as pd


def main():
    # Read January 2023 BTS flight data
    flights = pd.read_csv('data/bts2023.1.csv')

    # Keep only flights departing from SEA
    is_sea = flights['ORIGIN'] == 'SEA'
    flights = flights[is_sea]

    # Remove cancelled and diverted flights
    not_cancelled = flights['CANCELLED'] == 0
    not_diverted = flights['DIVERTED'] == 0
    flights = flights[not_cancelled & not_diverted]

    # Check departure delay data
    print('Number of SEA flights after filtering:')
    print(len(flights))

    print('Number of non-missing DEP_DELAY values:')
    print(flights['DEP_DELAY'].count())

    # Create delay label
    # A flight is delayed if departure delay is at least 15 minutes
    flights['delayed'] = flights['DEP_DELAY'] >= 15

    print('Number of delayed flights:')
    print(flights['delayed'].sum())

    print('Delay rate:')
    print(flights['delayed'].mean())

    # Convert CRS_DEP_TIME from HHMM format
    # Example: 1303 -> hour 13, minute 3
    flights['departure_hour'] = flights['CRS_DEP_TIME'] // 100
    flights['departure_minute'] = flights['CRS_DEP_TIME'] % 100

    # Combine flight date, hour, and minute
    flights['scheduled_departure_local'] = (
        pd.to_datetime(
            flights['FL_DATE'],
            format='%m/%d/%Y %I:%M:%S %p'
        )
        + pd.to_timedelta(
            flights['departure_hour'],
            unit='h'
        )
        + pd.to_timedelta(
            flights['departure_minute'],
            unit='m'
        )
    )

    # Convert Seattle local time to UTC
    flights['scheduled_departure_utc'] = (
        flights['scheduled_departure_local']
        .dt.tz_localize('America/Los_Angeles')
        .dt.tz_convert('UTC')
    )

    # Create a unique ID for each flight
    flights['flight_id'] = (
        flights['FL_DATE'].astype(str)
        + '_'
        + flights['OP_UNIQUE_CARRIER'].astype(str)
        + '_'
        + flights['OP_CARRIER_FL_NUM'].astype(str)
        + '_'
        + flights['ORIGIN'].astype(str)
        + '_'
        + flights['DEST'].astype(str)
        + '_'
        + flights['CRS_DEP_TIME'].astype(str)
    )

    # Keep only columns needed for later analysis and weather merge
    cleaned_flights = flights[[
        'flight_id',
        'FL_DATE',
        'DAY_OF_WEEK',
        'OP_UNIQUE_CARRIER',
        'OP_CARRIER_FL_NUM',
        'ORIGIN',
        'DEST',
        'CRS_DEP_TIME',
        'scheduled_departure_local',
        'scheduled_departure_utc',
        'DEP_DELAY',
        'delayed',
        'CRS_ELAPSED_TIME',
        'DISTANCE'
    ]]

    # Rename columns using team's snake_case convention
    cleaned_flights = cleaned_flights.rename(columns={
        'FL_DATE': 'flight_date',
        'DAY_OF_WEEK': 'day_of_week',
        'OP_UNIQUE_CARRIER': 'airline',
        'OP_CARRIER_FL_NUM': 'flight_number',
        'ORIGIN': 'origin',
        'DEST': 'destination',
        'CRS_DEP_TIME': 'scheduled_departure_time',
        'DEP_DELAY': 'departure_delay',
        'CRS_ELAPSED_TIME': 'scheduled_duration',
        'DISTANCE': 'distance'
    })

    # Check whether each flight_id is unique
    print('Duplicate flight IDs:')
    print(cleaned_flights['flight_id'].duplicated().sum())

    print('Final number of flights:')
    print(len(cleaned_flights))

    print('Final cleaned flight data:')
    print(cleaned_flights)

    # Save cleaned flight dataset
    cleaned_flights.to_csv(
        'data/clean_flights_sea_2023_01.csv',
        index=False
    )


if __name__ == '__main__':
    main()