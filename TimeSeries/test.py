import pandas as pd
from timeseries import LRTimeSeries

if __name__=='__main__':
    data = pd.read_csv('price_data.csv')

    lrts = LRTimeSeries(100, 5)
    lrts.train(data.price[:35000])
    print(lrts.predict())
