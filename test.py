import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data1 = {
    'bmi': [25, 30, 22, 28, 35, 24, 31, 27],
    'charges': [1000, 1500, 800, 1200, 2000, 900, 1600, 1300],
    'smoker': ['yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no']
}
insurance_data = pd.DataFrame(data1)

sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
plt.title("BMI vs Insurance Charges by Smoking Status")
plt.xlabel("BMI")
plt.ylabel("Insurance Charges")
plt.show()
data = {
    'date': ['2017-01-01', '2017-03-01', '2017-05-01', '2017-07-01', '2017-09-01', '2017-11-01', '2018-01-01'],
    'value': [10, 20, 15, 30, 10, 20, 35]
}
spotify_data = pd.DataFrame(data)

spotify_data['date'] = pd.to_datetime(spotify_data['date'])

sns.lineplot(x='date', y='value', data=spotify_data)

plt.title("Daily Global Streams of Popular Songs (2017-2018)")
plt.xlabel("Date")
plt.ylabel("Streams")

plt.show()