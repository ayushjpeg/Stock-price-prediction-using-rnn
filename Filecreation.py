import yfinance as yf

tickers = ['TTM']
start_date = "2022-01-01"   # YYYY-MM-DD
end_date = "2023-03-28"


df_train = yf.download(tickers, start=start_date, end=end_date)
df_train.to_csv("data/"+tickers[0].lower()+start_date[2:4]+"-"+end_date[2:4]+".csv")
print("File Created")
