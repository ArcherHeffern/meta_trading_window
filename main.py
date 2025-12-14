from itertools import chain
from statistics import stdev
from dataclasses import dataclass
from typing import Any, Iterable
from matplotlib import pyplot as plt
from datetime import datetime
from collections import defaultdict
from matplotlib.figure import Figure
import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta

# TODO: Non linear regression

FILENAME = "./meta_stock.csv"
PLOT_ROW_LIMIT = 4
NUM_SUBPLOT_COLUMNS = 4


@dataclass
class TradingWindowMetrics:
    # Within trading window
    zscores: Iterable[float]
    avg_zscore: float
    avg_extra_money: float
    # Month before and after trading window, excluding trading window
    standard_deviation: float


def predict(model: LinearRegression, x: list[float]) -> list[float]:
    predictions: list[float] = model.predict(np.array(list([v] for v in x)))  # type: ignore
    return list(predictions)


def is_trading_window(month: int) -> bool:
    return month in [2, 5, 8, 11]


def get_cur_subplot(axs: Any, trading_windows_processed: int, num_subplot_columns: int):
    # Start at 0
    x = trading_windows_processed % PLOT_ROW_LIMIT
    y = trading_windows_processed // PLOT_ROW_LIMIT
    return axs[x, y]


def process_trading_window(
    cur_plot: Any,
    start_of_month: datetime,
    group_by_month: defaultdict[datetime, list[tuple[datetime, float]]],
) -> TradingWindowMetrics:
    cur_plot.set_title(
        f"{start_of_month.day}/{start_of_month.month}/{start_of_month.year}"
    )
    print(f"__Trading Window: {start_of_month}__")
    prev_month = start_of_month - relativedelta(months=1)
    next_month = start_of_month + relativedelta(months=1)
    prev_month_data = group_by_month[prev_month]
    cur_month_data = group_by_month[start_of_month]
    next_month_data = group_by_month[next_month]

    # Find the line of best fit
    x = np.empty(len(prev_month_data) + len(next_month_data))
    y = np.empty(len(prev_month_data) + len(next_month_data))

    start_month = datetime(year=prev_month.year, month=prev_month.month, day=1)
    for i, (date, price) in enumerate(chain(prev_month_data, next_month_data)):
        dt = date - start_month
        x[i] = dt.days
        y[i] = price
    X = x.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)

    # Plot Trading Window
    trading_window_y = np.empty(len(cur_month_data))
    trading_window_x = np.empty(len(cur_month_data))
    for i, (day, price) in enumerate(cur_month_data):
        trading_window_y[i] = price
        trading_window_x[i] = (day - prev_month).days
    cur_plot.plot(trading_window_x, trading_window_y)

    # Plot Prev and Next Month
    cur_plot.plot(x, y)
    # Plot Line of best fit for prev and next month
    total_days = ((next_month + relativedelta(months=1)) - prev_month).days
    cur_plot.plot([0, total_days], list(predict(reg, [0, total_days])))

    # Find standard_deviation and mean for Non Trading Window
    non_window_differences = np.empty(len(prev_month_data) + len(next_month_data))
    for i, (date, price) in enumerate(chain(prev_month_data, next_month_data)):
        dt = date - start_month
        diff = price - predict(reg, [dt.days])[0]
        non_window_differences[i] = diff

    std = stdev(non_window_differences)
    mean = np.average(non_window_differences)

    # Find differences of Trading Window
    window_differences = np.empty(len(cur_month_data))
    for i, (date, price) in enumerate(cur_month_data):
        dt = date - start_month
        diff = price - predict(reg, [dt.days])[0]
        window_differences[i] = diff

    # Find average Z-score
    zscores = (window_differences - mean) / std
    avg_zscore: float = float(np.mean(zscores))
    avg_extra_money = avg_zscore * std
    print(f"Avg Z-score: {avg_zscore}")
    print(f"Avg Extra Profit: {avg_extra_money}")
    print(f"Zscores: {zscores}")
    return TradingWindowMetrics(zscores, avg_zscore, avg_extra_money, std)


def get_data() -> tuple[defaultdict[datetime, list[tuple[datetime, float]]], int]:
    group_by_month: defaultdict[datetime, list[tuple[datetime, float]]] = defaultdict(
        list
    )
    num_trading_windows = 0
    with open(FILENAME, newline="") as csvfile:
        next(csvfile)
        for row in csvfile:
            d, p = row.split(",")
            try:
                d = datetime.strptime(d, "%m/%d/%Y %H:%M:%S")
            except:
                print(d)
                continue
            p = float(p)
            start_of_month = datetime(d.year, d.month, 1)
            group_by_month[start_of_month].append((d, p))
            if is_trading_window(start_of_month.month):
                num_trading_windows += 1
    return group_by_month, num_trading_windows


def setup_plot(num_trading_windows: int) -> tuple[Figure, Any]:
    num_subplot_rows = num_trading_windows // NUM_SUBPLOT_COLUMNS
    if (
        num_trading_windows / NUM_SUBPLOT_COLUMNS
    ) != num_trading_windows // NUM_SUBPLOT_COLUMNS:
        num_subplot_rows += 1
    num_subplot_rows = min(num_subplot_rows, PLOT_ROW_LIMIT)
    return plt.subplots(NUM_SUBPLOT_COLUMNS, num_subplot_rows)


def main():
    # Get all prices in a month
    group_by_month, num_trading_windows = get_data()
    fig, axs = setup_plot(num_trading_windows)

    # Get average price difference before and after trading window
    trading_windows_processed = 0
    trading_window_metrics: dict[datetime, TradingWindowMetrics] = {}
    for start_of_month in group_by_month.keys():
        if not is_trading_window(start_of_month.month):
            continue

        if trading_windows_processed >= PLOT_ROW_LIMIT * NUM_SUBPLOT_COLUMNS:
            break
        cur_plot = get_cur_subplot(axs, trading_windows_processed, NUM_SUBPLOT_COLUMNS)
        trading_window_metrics[start_of_month] = process_trading_window(
            cur_plot, start_of_month, group_by_month
        )
        trading_windows_processed += 1

    # Meta metrics computation
    avg_avg_zscore = sum(m.avg_zscore for m in trading_window_metrics.values()) / len(
        trading_window_metrics
    )
    avg_avg_extra_money = sum(
        m.avg_extra_money for m in trading_window_metrics.values()
    ) / len(trading_window_metrics)
    print("__META METRICS__")
    print(f"avg_avg_zscores: {avg_avg_zscore}")
    print(f"avg_avg_extra_money: {avg_avg_extra_money}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
