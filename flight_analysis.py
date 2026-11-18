import pandas as pd


def main():
    # --------------------------------------------------
    # 1. Read all 12 months of 2023 flight data
    # --------------------------------------------------
    monthly_flights = []

    for month in range(1, 13):
        file_name = 'data/bts2023.' + str(month) + '.csv'
        month_data = pd.read_csv(file_name)
        monthly_flights.append(month_data)

    # Project extension: combine the 12 DataFrames
    flights = pd.concat(monthly_flights, ignore_index=True)

    print('Number of all 2023 flight records:')
    print(len(flights))

    # --------------------------------------------------
    # 2. Keep only SEA, DFW, and JFK departures
    # --------------------------------------------------
    is_sea = flights['ORIGIN'] == 'SEA'
    is_dfw = flights['ORIGIN'] == 'DFW'
    is_jfk = flights['ORIGIN'] == 'JFK'

    target_airports = is_sea | is_dfw | is_jfk
    flights = flights[target_airports]

    print('Number of target-airport flights before filtering:')
    print(len(flights))

    print('SEA flights before filtering:')
    print((flights['ORIGIN'] == 'SEA').sum())

    print('DFW flights before filtering:')
    print((flights['ORIGIN'] == 'DFW').sum())

    print('JFK flights before filtering:')
    print((flights['ORIGIN'] == 'JFK').sum())

    # --------------------------------------------------
    # 3. Remove cancelled and diverted flights
    # --------------------------------------------------
    not_cancelled = flights['CANCELLED'] == 0
    not_diverted = flights['DIVERTED'] == 0

    flights = flights[not_cancelled & not_diverted]

    print('Number of flights after filtering:')
    print(len(flights))

    print('Number of non-missing DEP_DELAY values:')
    print(flights['DEP_DELAY'].count())

    # --------------------------------------------------
    # 4. Create delay label
    # Delay = at least 15 minutes late
    # --------------------------------------------------
    flights['delayed'] = flights['DEP_DELAY'] >= 15

    print('Number of delayed flights:')
    print(flights['delayed'].sum())

    print('Overall delay rate:')
    print(flights['delayed'].mean())

    # Airport-specific delay rates
    sea = flights['ORIGIN'] == 'SEA'
    dfw = flights['ORIGIN'] == 'DFW'
    jfk = flights['ORIGIN'] == 'JFK'

    print('SEA delay rate:')
    print(flights.loc[sea, 'delayed'].mean())

    print('DFW delay rate:')
    print(flights.loc[dfw, 'delayed'].mean())

    print('JFK delay rate:')
    print(flights.loc[jfk, 'delayed'].mean())

    # --------------------------------------------------
    # 5. Split CRS_DEP_TIME into hour and minute
    # Example: 1303 -> 13:03
    # --------------------------------------------------
    flights['departure_hour'] = (
        flights['CRS_DEP_TIME'] // 100
    )

    flights['departure_minute'] = (
        flights['CRS_DEP_TIME'] % 100
    )

    # --------------------------------------------------
    # 6. Create local scheduled departure datetime
    # Project extension
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 7. Convert each airport's local time to UTC
    #
    # SEA = Pacific Time
    # DFW = Central Time
    # JFK = Eastern Time
    #
    # Project extension
    # --------------------------------------------------
    sea_times = flights.loc[
        flights['ORIGIN'] == 'SEA',
        'scheduled_departure_local'
    ]

    dfw_times = flights.loc[
        flights['ORIGIN'] == 'DFW',
        'scheduled_departure_local'
    ]

    jfk_times = flights.loc[
        flights['ORIGIN'] == 'JFK',
        'scheduled_departure_local'
    ]

    sea_utc = (
        sea_times
        .dt.tz_localize(
            'America/Los_Angeles',
            ambiguous=False,
            nonexistent='shift_forward'
        )
        .dt.tz_convert('UTC')
    )

    dfw_utc = (
        dfw_times
        .dt.tz_localize(
            'America/Chicago',
            ambiguous=False,
            nonexistent='shift_forward'
        )
        .dt.tz_convert('UTC')
    )

    jfk_utc = (
        jfk_times
        .dt.tz_localize(
            'America/New_York',
            ambiguous=False,
            nonexistent='shift_forward'
        )
        .dt.tz_convert('UTC')
    )

    flights['scheduled_departure_utc'] = pd.concat([
        sea_utc,
        dfw_utc,
        jfk_utc
    ]).sort_index()

    # --------------------------------------------------
    # 8. Create unique flight ID
    # --------------------------------------------------
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

    # 9. Keep variables needed for the project
    cleaned_flights = flights[[
        'flight_id',
        'YEAR',
        'MONTH',
        'DAY_OF_WEEK',
        'FL_DATE',
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

    # 10. Rename columns to team snake_case format
    # Project extension
    cleaned_flights = cleaned_flights.rename(columns={
        'YEAR': 'year',
        'MONTH': 'month',
        'DAY_OF_WEEK': 'day_of_week',
        'FL_DATE': 'flight_date',
        'OP_UNIQUE_CARRIER': 'airline',
        'OP_CARRIER_FL_NUM': 'flight_number',
        'ORIGIN': 'origin',
        'DEST': 'destination',
        'CRS_DEP_TIME': 'scheduled_departure_time',
        'DEP_DELAY': 'departure_delay',
        'CRS_ELAPSED_TIME': 'scheduled_duration',
        'DISTANCE': 'distance'
    })

    # 11. Final checks
    print('Duplicate flight IDs:')
    print(cleaned_flights['flight_id'].duplicated().sum())

    print('Final number of flights:')
    print(len(cleaned_flights))

    print('Final SEA flights:')
    print((cleaned_flights['origin'] == 'SEA').sum())

    print('Final DFW flights:')
    print((cleaned_flights['origin'] == 'DFW').sum())

    print('Final JFK flights:')
    print((cleaned_flights['origin'] == 'JFK').sum())

    # 12. Save cleaned 2023 flight dataset
    cleaned_flights.to_csv(
        'data/clean_flights_2023.csv',
        index=False
    )

    print('Saved:')
    print('data/clean_flights_2023.csv')


if __name__ == '__main__':
    main()