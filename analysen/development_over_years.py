import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.ticker as mtick
import re

DATA_DIR = Path(__file__).parent.parent / "daten"

df = pd.DataFrame(columns=["date"]).set_index("date")

for counter_file in sorted(DATA_DIR.glob("*.csv")):
    name = (
        re.sub(r"^\d+_", "", counter_file.stem)
        .replace("_", " ")
        .replace(" rad", "")
        .replace(" kpl", "")
        .strip()
        .title()
    )
    if counter_file.stat().st_size == 0:
        print(f"Skipping {name} (is empty)")
        continue

    print(f"Loading {name}...")
    counter_df = pd.read_csv(
        counter_file, parse_dates=["Datum"], date_format="%d.%m.%Y"
    )
    counter_df.columns = ["date", name]
    counter_df = counter_df.set_index("date")
    df = df.join(counter_df, how="outer")


# breakpoint()


def drop_outliers(df: pd.DataFrame) -> None:
    """
    Drop outliers using an IQR filter.

    See OPTION 3 in <https://stackoverflow.com/a/69001342/3077972>.
    """
    # limits to a (float), b (int) and e (timedelta)
    cols = df.select_dtypes("number").columns
    df_sub = df.loc[:, cols]

    # IQR filter: within 2.22 IQR (equiv. to z-score < 3)
    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(
        0.25, numeric_only=False
    )
    lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22

    # replace outliers with nan
    df.loc[:, cols] = df_sub.where(lim, np.nan)


drop_outliers(df)

# breakpoint()

df = df.reindex(
    pd.date_range(
        f"{df.index.min().year}-01-01", f"{df.index.max().year}-12-31", freq="D"
    )
)

# breakpoint()


def mean_with_min_count(group):
    assert len(group.shape) == 1
    return group.mean() if group.count() > group.shape[0] * 0.9 else pd.NA


# df = df.groupby(df.index.year).mean().dropna(axis='columns')
df = df.groupby(df.index.year).agg(mean_with_min_count)

# breakpoint()

df = df.dropna(axis="index", how="all").dropna(axis="columns", how="all")

# breakpoint()

first_entries = df.apply(
    lambda s: (
        s.loc[s.first_valid_index()] if s.first_valid_index() is not None else pd.NA
    )
)
development_over_years = (df / first_entries).replace(pd.NA, None)
development_over_years = development_over_years.dropna(
    axis="columns", thresh=development_over_years.shape[0] * 0.5
)
ax = (development_over_years * 100).plot(
    # grid=True,
    legend=True,
    xlabel="Jahr",
    ylabel="Entwicklung im Vergleich zum ersten Jahr der Zählstation",
    marker="o",
    title="Qualitative Entwicklung der Zahlen nach relevanter Zählstelle",
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
print(development_over_years)
plt.show()
