# ocean_tools/data_handling/availability.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def data_availability_analysis(urls_directory: str, monthly: bool = False, plot: bool = True) -> None:
    """
    Analyzes file availability from a text file containing URLs.

    This function reads a text file of URLs, extracts dates from the file names,
    groups the dates by month, and calculates the number of available files versus the
    ideal number (based on the number of days in each month, unless monthly is True).
    Optionally, it plots the number of available and missing files per month.

    Parameters
    ----------
    urls_filepath : str
        Path to the text file containing URLs (one URL per line). The file names are expected
        to contain a date in the format YYYYMMDD after a dot.
    monthly : bool, optional
        If True, assumes each month should have one file (instead of the total number of days in the month).
        Default is False.
    plot : bool, optional
        If True, displays a bar chart of file availability. Default is True.

    Returns
    -------
    None
    """
    with open(urls_directory, "r") as file:
        file_urls = file.read()

    url_list = list(filter(None, map(str.strip, file_urls.split("\n"))))
    file_names = [url.split("/")[-1] for url in url_list]

    dates = [file_name.split(".")[1][:8] for file_name in file_names]
    df = pd.DataFrame(dates, columns=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.sort_values(by="Date", inplace=True)

    df_monthly = df.groupby(df["Date"].dt.to_period("M")).agg(
        files_available=pd.NamedAgg(column="Date", aggfunc="count")
    )

    date_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq='MS')
    df_monthly = df_monthly.reindex(date_range.to_period('M'))
    df_monthly["files_available"] = df_monthly["files_available"].fillna(0)

    df_monthly["total_days"] = df_monthly.index.map(lambda x: 1 if monthly else x.daysinmonth)
    df_monthly["missing_days"] = df_monthly["total_days"] - df_monthly["files_available"]

    total_files = df_monthly['files_available'].sum()
    ideal_files = df_monthly['total_days'].sum()

    print(f"Found {total_files} files out of idealy {ideal_files} files.")
    print(f"Completion Rate: {total_files / ideal_files * 100:.2f}%")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

    if plot:
        df_monthly["files_available"].plot(kind="bar", figsize=(35, 5), color="green", edgecolor="black")
        df_monthly["missing_days"].plot(kind="bar", figsize=(35, 5), color="red", edgecolor="black", bottom=df_monthly["files_available"])
        plt.title(f"Availability over {os.path.basename(urls_directory)}")
        plt.xlabel("Year-Month")
        plt.ylabel("Number of files")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
