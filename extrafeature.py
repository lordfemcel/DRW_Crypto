def add_imbalance_feature(df):
    df['imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-6)
    return df
