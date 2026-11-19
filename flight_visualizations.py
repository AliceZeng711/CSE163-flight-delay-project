import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_departure_period(time):
    """Return the departure period for a scheduled HHMM time."""
    hour = time // 100

    if hour < 6:
        return 'Night'
    elif hour < 12:
        return 'Morning'
    elif hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'


def plot_delay_rate_by_airport(flights):
    """Plot the 2023 delay rate for SEA, DFW, and JFK."""
    flights['delay_percent'] = flights['delayed'] * 100

    sns.catplot(
        data=flights,
        x='origin',
        y='delay_percent',
        kind='bar',
        errorbar=None
    )

    plt.xlabel('Airport')
    plt.ylabel('Delay Rate (%)')
    plt.title('2023 Departure Delay Rate by Airport')

    plt.savefig(
        'delay_rate_by_airport.png',
        bbox_inches='tight'
    )

    plt.close()


def plot_delay_rate_by_month(flights):
    """Plot monthly delay rates for SEA, DFW, and JFK."""
    flights['delay_percent'] = flights['delayed'] * 100

    sns.relplot(
        data=flights,
        x='month',
        y='delay_percent',
        hue='origin',
        kind='line',
        marker='o',
        errorbar=None
    )

    plt.xlabel('Month')
    plt.ylabel('Delay Rate (%)')
    plt.title('2023 Monthly Departure Delay Rate by Airport')

    plt.savefig(
        'delay_rate_by_month.png',
        bbox_inches='tight'
    )

    plt.close()


def plot_delay_rate_by_departure_period(flights):
    """Plot delay rate by scheduled departure period."""
    flights['departure_period'] = (
        flights['scheduled_departure_time'].apply(
            get_departure_period
        )
    )

    flights['delay_percent'] = flights['delayed'] * 100

    sns.catplot(
        data=flights,
        x='departure_period',
        y='delay_percent',
        hue='origin',
        kind='bar',
        order=[
            'Night',
            'Morning',
            'Afternoon',
            'Evening'
        ],
        errorbar=None
    )

    plt.xlabel('Scheduled Departure Period')
    plt.ylabel('Delay Rate (%)')
    plt.title('2023 Delay Rate by Scheduled Departure Period')

    plt.savefig(
        'delay_rate_by_departure_period.png',
        bbox_inches='tight'
    )

    plt.close()


def main():
    flights = pd.read_csv(
        'data/clean_flights_2023.csv'
    )

    plot_delay_rate_by_airport(flights)
    plot_delay_rate_by_month(flights)
    plot_delay_rate_by_departure_period(flights)


if __name__ == '__main__':
    main()