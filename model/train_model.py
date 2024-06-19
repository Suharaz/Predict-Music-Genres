from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os
import pandas as pd
import extract_features as ef


dataset_path = "data_path_here"
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
audio_features = []
labels = []

# Duyệt qua từng thể loại và tập tin âm thanh để trích xuất đặc trưng
for genre in genres:
    print(f"Extracting features for genre: {genre}")
    genre_path = os.path.join(dataset_path, genre)
    for track in os.listdir(genre_path):
        track_name = os.path.join(genre_path, track)
        features = ef.extract_features(track_name)
        print(track_name)
        if features is not None:
            audio_features.append(features)
            labels.append(genre)
columns = [
    'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean',
    'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
    'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var',
    'tempo'
] + [f'mfcc{i+1}_mean' for i in range(20)] + [f'mfcc{i+1}_var' for i in range(20)]

df = pd.DataFrame(audio_features, columns=columns)
df["label"]=labels         
# Chuẩn bị dữ liệu
X = df.drop("label", axis=1)
y = df["label"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mã hóa nhãn
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_test = labelencoder.transform(y_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Định nghĩa các thông số cho RandomizedSearchCV
param_dist_svm = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf'],
}

# Tạo bộ phân loại SVM và đối tượng RandomizedSearchCV
svm = SVC(random_state=42, probability=True)
random_search_svm = RandomizedSearchCV(
    svm, param_distributions=param_dist_svm, n_iter=50,
    scoring='accuracy', n_jobs=-1, random_state=42
)

# Huấn luyện mô hình SVM
random_search_svm.fit(X_train, y_train)

# Đánh giá mô hình SVM với các thông số tốt nhất trên tập kiểm tra
best_svm = random_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Đánh giá mô hình SVM trên tập huấn luyện
y_train_pred_svm = best_svm.predict(X_train)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)

# Tính toán các chỉ số đánh giá khác
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)

# In các kết quả đánh giá
print("Train SVM Accuracy:", train_accuracy_svm)
print("Test SVM Accuracy:", test_accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1 Score:", f1_svm)
print("Confusion Matrix:\n", conf_matrix_svm)
print("Classification Report:\n", class_report_svm)
