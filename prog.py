import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df = pd.read_csv('data.txt', sep=',')

df_new = pd.read_csv('test.txt', sep=',')

print(df.head())
print(df_new.head())

# label_encoder = LabelEncoder()

# df['targy'] = label_encoder.fit_transform(df['targy'])
# df['ido'] = label_encoder.fit_transform(df['ido'])
# df['felado'] = label_encoder.fit_transform(df['felado'])
# df['osztaly'] = label_encoder.fit_transform(df['osztaly']) 

# df_new['targy'] = label_encoder.transform(df_new['targy'])
# df_new['ido'] = label_encoder.transform(df_new['ido'])
# df_new['felado'] = label_encoder.transform(df_new['felado'])


# X = df.drop(columns=['osztaly'])
# y = df['osztaly']


# # Képzés és tesztelés felosztása
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Modell tréningezése
# model = GaussianNB()
# model.fit(X_train, y_train)


# # A modellel előrejelzést készítünk
# X_new = df_new.drop(columns=['osztaly'])  # Az 'osztaly' oszlop itt nincs
# y_prob = model.predict_proba(X_new)  # Ki akarjuk kérni a valószínűségeket

# # A valószínűségeket és a predikciókat kiíratjuk
# for i, prob in enumerate(y_prob):
#     predicted_class = label_encoder.inverse_transform([prob.argmax()])[0]
#     print(f"Row {i+1} - Predicted class: {predicted_class}, Probability: {prob}")