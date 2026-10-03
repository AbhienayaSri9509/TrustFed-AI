from pathlib import Path
import pandas as pd
def load_table(path):
    p=Path(path)
    if p.suffix.lower()=='.csv': return pd.read_csv(p)
    if p.suffix.lower() in ('.xlsx','.xls'): return pd.read_excel(p)
    raise ValueError(f'Unsupported file: {p}')
def inspect_dataset(path):
    df=load_table(path)
    print('Shape:',df.shape)
    print('Columns:',list(df.columns))
    print('\nMissing values:')
    print(df.isna().sum().sort_values(ascending=False).head(20))
    return df
if __name__=='__main__': inspect_dataset('data/raw/UNSW_NB15_training-set.xlsx')
