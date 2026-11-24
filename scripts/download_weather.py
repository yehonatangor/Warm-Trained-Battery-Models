"""
Download hourly ECCC weather (Dec-Mar, winters 2015/16 to 2024/25) for three stations
and join it into data/raw/weather/weather_winters_2015_2025.csv
Finished monthly files are skipped, so a stopped run can be restarted
Never add an empty time= parameter to the URL: the server then returns empty rows
"""

import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from warmtrained import config as cfg

STATIONS = {"LONDON_CS": 10999, "TORONTO_INTL_A": 51459, "EDMONTON_INTL_A": 50149}
FIRST_YEARS = range(2015, 2025)  # winter 2015/16 starts in 2015
URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv"
    "&stationID={sid}&Year={y}&Month={m}&Day=1&timeframe=1&submit=Download+Data"
)
DIR = cfg.DATA_RAW / "weather"
OUT = DIR / "weather_winters_2015_2025.csv"
MIN_BYTES = 50_000  # real month is far bigger; error page is smaller


def winter_months():
    for y in FIRST_YEARS:
        yield (y, 12)
        for m in (1, 2, 3):
            yield (y + 1, m)


def month_file(name, y, m):
    return DIR / name / f"{name}_{y}-{m:02d}.csv"


def fetch(url, path):
    tmp = path.with_suffix(".part")  # partial download never looks finished
    for attempt in range(1, 5):  # server drops connections sometimes
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
            return
        except OSError:
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {path.name}")


def download():
    for name, sid in STATIONS.items():
        (DIR / name).mkdir(parents=True, exist_ok=True)
        for y, m in winter_months():
            path = month_file(name, y, m)
            if path.exists() and path.stat().st_size > MIN_BYTES:
                continue
            fetch(URL.format(sid=sid, y=y, m=m), path)
            print(path.name, flush=True)
            time.sleep(1)


def join():
    cols = {"Date/Time (LST)": "time", "Temp (°C)": "T", "Temp Flag": "T_flag"}
    parts = [
        pd.read_csv(month_file(name, y, m), encoding="utf-8-sig", usecols=list(cols))
        .rename(columns=cols)
        .assign(station=name)
        for name in STATIONS
        for y, m in winter_months()
    ]
    w = (
        pd.concat(parts, ignore_index=True)[["station", "time", "T", "T_flag"]]
        .sort_values(["station", "time"])
        .reset_index(drop=True)
    )
    w.to_csv(OUT, index=False)
    print(w.groupby("station").agg(hours=("time", "size"), missing_T=("T", lambda t: t.isna().sum())))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    download()
    join()