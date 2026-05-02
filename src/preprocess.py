import pandas as pd

def load_data(filepath):
    """Load and preprocess"""

    df = pd.read_csv(
    filepath,
    sep=r"\s+",
    header=None
    )

    df = df.dropna(axis=1, how='all')

    # Add column names
    columns = ['unit', 'cycle', 'op1', 'op2', 'op3']

    for i in range(1, 22):
        columns.append(f'sensor_{i}')

    df.columns = columns

    return df

def create_rul(df):
    """Create Remaining Useful Life (RUL) feature"""

    max_cycle = df.groupby('unit')['cycle'].max()

    df = df.merge(
        max_cycle.rename('max_cycle'),
        on='unit'
    )

    df['RUL'] = df['max_cycle'] - df['cycle']

    return df


def get_sensor_columns(df):
    """Return sensor feature cols"""

    sensor_columns = [
    col for col in df.columns
    if 'sensor' in col
    ]

    return sensor_columns

def clip_rul(df,max_rul=125):
    """Clip RUL values(large vals)"""

    df['RUL'] = df['RUL'].clip(upper=max_rul)

    return df