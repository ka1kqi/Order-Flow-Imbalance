import pandas as pd
import numpy as np
import statistics
from sklearn.decomposition import PCA


def load_data(filepath):
    #filepath: str
    df = pd.DataFrame(pd.read_csv(filepath))
    return df

def best_level_ofi(pdf):
    #df: pd.DataFrame
    df = pdf.set_index('ts_event')
    best_level_ofi = pd.DataFrame(index=df.index)

    df['bid_px_00_prev'] = df['bid_px_00'].shift(1)
    df['ask_px_00_prev'] = df['ask_px_00'].shift(1)
    df['bid_sz_00_prev'] = df['bid_sz_00'].shift(1)
    df['ask_sz_00_prev'] = df['ask_sz_00'].shift(1)

    conditions_bid = [
        df['bid_px_00'] > df['bid_px_00_prev'],
        df['bid_px_00'] == df['bid_px_00_prev'],
        df['bid_px_00'] < df['bid_px_00_prev']
    ]
    choices_bid = [
        df['bid_sz_00'],
        df['bid_sz_00'] - df['bid_sz_00_prev'],
        df['bid_sz_00'] * -1
    ]

    conditions_ask = [
        df['ask_px_00'] > df['ask_px_00_prev'],
        df['ask_px_00'] == df['ask_px_00_prev'],
        df['ask_px_00'] < df['ask_px_00_prev']
    ]
    choices_ask = [
        df['ask_sz_00'] * -1,
        df['ask_sz_00'] - df['ask_sz_00_prev'],
        df['ask_sz_00']
    ]

    df['OF_bid'] = np.select(conditions_bid, choices_bid)
    df['OF_ask'] = np.select(conditions_ask, choices_ask)
    best_level_ofi['BEST_LEVEL_OFI'] = df['OF_bid'] - df['OF_ask']

    return best_level_ofi

def multi_level_ofi(pdf):
  #pdf: pd.DataFrame
  df = pd.DataFrame()
  df = pdf.set_index('ts_event')
  multi_ofi = pd.DataFrame(index=df.index)

  depths = []

  for level in range(0,10):
    bid_px = f'bid_px_{level:02d}'
    ask_px = f'ask_px_{level:02d}'
    bid_sz = f'bid_sz_{level:02d}'
    ask_sz = f'ask_sz_{level:02d}'

    depths.extend(df[bid_sz]+df[ask_sz])

    df[f'{bid_px}_prev'] = df[bid_px].shift(1)
    df[f'{ask_px}_prev'] = df[ask_px].shift(1)
    df[f'{bid_sz}_prev'] = df[bid_sz].shift(1)
    df[f'{ask_sz}_prev'] = df[ask_sz].shift(1)

    conditions_bid = [
      df[bid_px] > df[f'{bid_px}_prev'],
      df[bid_px] == df[f'{bid_px}_prev'],
      df[bid_px] < df[f'{bid_px}_prev']
    ]
    choices_bid = [
      df[bid_sz],
      df[bid_sz] - df[f'{bid_sz}_prev'],
      df[bid_sz] * -1
    ]

    conditions_ask = [
      df[ask_px] > df[f'{ask_px}_prev'],
      df[ask_px] == df[f'{ask_px}_prev'],
      df[ask_px] < df[f'{ask_px}_prev']
    ]
    choices_ask = [
      df[ask_sz] * -1,
      df[ask_sz] - df[f'{ask_sz}_prev'],
      df[ask_sz]
    ]

    df[f'OF_bid_{level}'] = np.select(conditions_bid, choices_bid,default = 0)
    df[f'OF_ask_{level}'] = np.select(conditions_ask, choices_ask,default = 0)
    df[f'OFI_{level}'] = df[f'OF_bid_{level}'] - df[f'OF_ask_{level}']
    multi_ofi[f'OFI_{level}'] = df[f'OFI_{level}']

  avg_depth = statistics.mean(depths)
  multi_ofi_df = pd.DataFrame(index = multi_ofi.index)
  for level in range(0,10):
    #scale OFIs
    multi_ofi_df[f'ofi_{level}'] = multi_ofi[f'OFI_{level}'] / avg_depth
  return multi_ofi_df

def integrated_ofi(multi_ofi):
  #multi_ofi: pd.DataFrame (multilevel ofi computed above)
  df = multi_ofi
  ofi_cols = [f'ofi_{level}' for level in range(0,10)]
  X = df[ofi_cols].values
  pca = PCA(n_components = 1)
  pca.fit(X)
  w1 = pca.components_[0]
  w1_normalized = w1 / np.linalg.norm(w1)
  integrated_df = pd.DataFrame(index = df.index)
  #compute integrated ofi as a dot product between w1 normalized and ofi vector
  integrated_df['integrated_ofi'] = np.dot(X,w1_normalized)

  return integrated_df


df = load_data('first_25000_rows.csv')
best_level_df = best_level_ofi(df)
multi_level_df = multi_level_ofi(df)
integrated_ofi_df = integrated_ofi(multi_level_df)
#there is only one symbol in this so there is no cross impact ofi to compute.
