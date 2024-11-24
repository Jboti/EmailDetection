import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import pearsonr

df = pd.read_csv('data.txt', sep=',')

test_df = pd.read_csv('test.txt', sep=',')

label_encoder = LabelEncoder()

df['targy'] = label_encoder.fit_transform(df['targy'])
df['ido'] = label_encoder.fit_transform(df['ido'])
df['felado'] = label_encoder.fit_transform(df['felado'])
df['osztaly'] = label_encoder.fit_transform(df['osztaly']) 

test_df['targy'] = label_encoder.fit_transform(test_df['targy'])
test_df['ido'] = label_encoder.fit_transform(test_df['ido'])
test_df['felado'] = label_encoder.fit_transform(test_df['felado'])


skala = MinMaxScaler()
X = skala.fit_transform(df.iloc[:, :-1])
y = df['osztaly']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


modell = MultinomialNB()
modell.fit(X_train, y_train)


y_pred = modell.predict(X_test)
print("Osztályozási Jelentés:")
print(classification_report(y_test, y_pred))
print("Konfúziós Mátrix:")
print(confusion_matrix(y_test, y_pred))


korrelaciok = []
for oszlop in df.columns[:-1]:
    corr, _ = pearsonr(df[oszlop], df['osztaly'])
    korrelaciok.append((oszlop, corr))

korrelaciok = sorted(korrelaciok, key=lambda x: abs(x[1]), reverse=True)


X_test_new = skala.fit_transform(test_df)

probabilities = modell.predict_proba(X_test_new)

print("\nTeszt adatok spam valószínűségei:")
for i, (email, prob) in enumerate(zip(test_df.values, probabilities)):
    spam_prob = prob[1]  
    print(f"Email {i+1}: {spam_prob:.2%} valószínűséggel nem spam")
