# ══════════════════════════════════════════════════════════════
#  Resume Screening – Enhanced Training Pipeline
#  Run this once to generate: clf.pkl, tfidf.pkl, label_encoder.pkl
# ══════════════════════════════════════════════════════════════
import os, pickle, re
import numpy  as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing           import LabelEncoder
from sklearn.model_selection         import train_test_split
from sklearn.svm                     import LinearSVC
from sklearn.multiclass              import OneVsRestClassifier
from sklearn.metrics                 import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, 'Resume.csv')

# ── 1. Load data ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=['Resume_html'], errors='ignore')
print(f"Dataset shape : {df.shape}")
print(f"Categories    : {df['Category'].nunique()}")
print(df['Category'].value_counts().to_string())

# ── 2. Clean text ──────────────────────────────────────────────
def clean_resume(text: str) -> str:
    text = re.sub(r'http\S+\s?',  ' ', str(text))
    text = re.sub(r'RT|cc',       ' ', text)
    text = re.sub(r'#\S+\s?',     ' ', text)
    text = re.sub(r'@\S+',        ' ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',' ', text)
    text = re.sub(r'\s+',         ' ', text)
    return text.strip()

df['cleaned'] = df['Resume_str'].apply(clean_resume)

# ── 3. Encode labels ───────────────────────────────────────────
le          = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])
print(f"\nLabel classes : {list(le.classes_)}")

# ── 4. TF-IDF  (key improvements over original) ───────────────
stop_words = list(stopwords.words('english'))

tfidf = TfidfVectorizer(
    stop_words   = stop_words,
    sublinear_tf = True,
    max_features = 50_000,
    min_df       = 2,
    max_df       = 0.95,
    ngram_range  = (1, 2),
)

X = tfidf.fit_transform(df['cleaned'])
y = df['label']
print(f"\nFeature matrix : {X.shape}")

# ── 5. Stratified train/test split ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y
)
print(f"Train size : {X_train.shape[0]}   Test size : {X_test.shape[0]}")

# ── 6. Train LinearSVC ────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV
clf = CalibratedClassifierCV(OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000)), cv=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n{'─'*50}")
print(f"Test Accuracy  : {acc*100:.2f}%")
print(f"{'─'*50}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 7. Save artefacts ─────────────────────────────────────────
pickle.dump(tfidf, open(os.path.join(BASE_DIR, 'tfidf.pkl'),         'wb'))
pickle.dump(clf,   open(os.path.join(BASE_DIR, 'clf.pkl'),           'wb'))
pickle.dump(le,    open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'wb'))

print("\n✅ Saved: tfidf.pkl  clf.pkl  label_encoder.pkl")
print("   Run app with:  streamlit run app.py")