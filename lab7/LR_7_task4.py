import json
import numpy as np
from sklearn import covariance
from sklearn.cluster import KMeans
import yfinance as yf
# Завантаження фінансових даних
# Завантаження символів компаній
# Файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'
# Завантаження даних із файлу
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T
# Завантаження історичних даних
start_date = '2013-01-01'
# Початок періоду
end_date = '2017-12-31'
# Кінець періоду
quotes = []
valid_symbols = []
valid_names = []
for symbol, name in zip(symbols, names):
    print(f"Downloading data for {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if not data.empty: # Перевірка, чи є дані
        quotes.append(data)
        valid_symbols.append(symbol)
        valid_names.append(name)
    else:
        print(f"Warning: No data for {symbol}")

# Оновлені списки компаній
valid_symbols = np.array(valid_symbols)
valid_names = np.array(valid_names)
# Перевірка і обробка довжин даних
data_lengths = [len(quote) for quote in quotes]
print(f"Data lengths before trimming: {data_lengths}")
# Виключення компаній з дуже малою кількістю даних
min_days_required = 10
quotes, valid_names = zip( *[(quote, name) for quote, name in zip(quotes, valid_names) if len(quote) >= min_days_required])
if len(quotes) == 0:
    raise ValueError("No valid data available for analysis.")
# Усічення до однакової довжини
min_length = min(len(quote) for quote in quotes)
quotes = [quote.iloc[:min_length] for quote in quotes]
# Обробка котирувань
opening_quotes = np.array([quote['Open'].values for quote in quotes], dtype=np.float64)
closing_quotes = np.array([quote['Close'].values for quote in quotes], dtype=np.float64)

# Обчислення різниці котирувань
quotes_diff = closing_quotes - opening_quotes
X = quotes_diff.copy().T
# Запобігання діленню на нуль
std_dev = X.std(axis=0)
std_dev[std_dev == 0] = 1
X /= std_dev
# Перевірка форми X та усунення зайвих вимірів
X = np.squeeze(X)
if X.ndim != 2:
    raise ValueError(f"Unexpected shape of X: {X.shape}")

print(f"Shape of X: {X.shape}")
if X.shape[0] < 5 or X.shape[1] < 2:
    raise ValueError("Insufficient data for covariance calculation.")
# Побудова моделі графа
print("Training GraphicalLassoCV model...")
edge_model = covariance.GraphicalLassoCV()
# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Кластеризація даних з K Means
print("Performing KMeans clustering...")
n_clusters = 7
# Кластеризація з KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X.T)
# Вивід результатів
valid_names = np.array(valid_names, dtype=str)
# Вивід кластерів
print("\nClusters found:")
for i in range(n_clusters):
    cluster_members = valid_names[labels == i]
    print(f"Cluster {i + 1} ==> {', '.join(cluster_members)}")