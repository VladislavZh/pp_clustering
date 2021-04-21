import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
data = pd.read_csv('../mds20_cohortney/data/booking_challenge_tpp_labeled.csv')
print(data.columns)
data = data.sort_values(by=['user_id', 'checkin'])
print(data.head(10))

print('user_id')
plt.figure()
inout = data.groupby(['user_id']).agg({'diff_inout': ['mean', 'count']})
inout['diff_inout']['count'].plot.hist(bins=200, alpha=0.5)
plt.title('Histogram of number of trips by user_id')
plt.show()
print(inout['diff_inout']['count'].min())
print(inout['diff_inout']['count'].max())
print(inout['diff_inout']['count'].mean())
print(inout['diff_inout']['count'].quantile([0.1,0.25,0.5,0.75,0.9]))
print('city_id')
plt.figure()
inout = data.groupby(['city_id']).agg({'diff_inout': ['mean', 'count']})
inout['diff_inout']['count'].plot.hist(bins=1000, alpha=0.5)
plt.title('Histogram of number of trips by city_id')
plt.show()
print(inout['diff_inout']['count'].min())
print(inout['diff_inout']['count'].max())
print(inout['diff_inout']['count'].mean())
print(inout['diff_inout']['count'].quantile([0.1,0.25,0.5,0.75,0.9]))

