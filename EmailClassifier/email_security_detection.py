import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 讀取資料集
df = pd.read_csv("email_classifier/phishing_email.csv", encoding='latin-1')

# 前 5 筆資料 / 對應欄列和屬性 / 正確標籤數量
print()
print(df.head())
print()
print(df.shape)
print()
print(df.dtypes)
print()
print(df["label"].value_counts())

# 缺失檢查和清理
print()
print(df.isnull().sum())
df = df.dropna(subset=["label"])
print()
print(df.isnull().sum())

# 將標籤轉換為數值
df["label"] = df["label"].astype(int)

# 區分特徵和標籤
X, y = df["text_combined"], df["label"]
print()
print(X.head())
print()
print(y.head())

# 將資料集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.2, 
    random_state = 42,
    stratify = y
)

# 將文字資料轉換為 TF-IDF 特徵向量
vector = TfidfVectorizer()
X_train_tfidf = vector.fit_transform(X_train)
X_test_tfidf = vector.transform(X_test)

# 模型訓練
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# 模型評估
print()
print(classification_report(y_test, y_pred))
print()
print(confusion_matrix(y_test, y_pred))
print()

# 模型測試
while True:
    email = input("請輸入 Email 內容 ( 輸入 'exit' 結束 ) : ")

    if email.lower() == 'exit':
        break
    elif not email.strip():
        print("未輸入任何內容，請重新輸入。")
        continue

    email_tfidf = vector.transform([email])
    prediction = model.predict(email_tfidf)
    probabilities = model.predict_proba(email_tfidf)[0]
    probability_normal = probabilities[0]
    probability_malicious = probabilities[1]

    if prediction[0] == 1:
        print(f"這是惡意郵件，惡意機率 : {probability_malicious:.2%}")
    else:
        print(f"這是正常郵件，正常機率 : {probability_normal:.2%}")