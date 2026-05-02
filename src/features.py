def add_rolling_features(df,sensor_columns, window=5):
    """Add rolling mean and std features for each sensor"""

    for col in sensor_columns:

        rolling_col = f"{col}_ma{window}"

        df[rolling_col] = (df.groupby('unit')[col]
                           .rolling(window)
                           .mean()
                           .reset_index(level=0, drop=True)
                           )
    return df