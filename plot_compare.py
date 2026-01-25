import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('results/speedup.csv')
df["Project"] = df["Project"].str.replace('Quant', 'Quantitative Trading Analysis')
df["Project"] = df["Project"].str.replace('Aes', 'Cryptography Algorithms')
df["Project"] = df["Project"].str.replace('TinyXML', 'Fast XML Parsing')


plt.figure(figsize=(10, 6))
plt.title('CPU Time Comparison by Project and Attempted Optimization Method')
sns.barplot(x='Project', y='Total CPU Time', hue='Code Iteration', data=df, palette='mako')

plt.savefig('results/speedup.png')
