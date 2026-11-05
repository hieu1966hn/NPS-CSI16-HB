import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

class AnimePredictor:
    def __init__(self, csv_path='anime-filtered.csv'):
        self.csv_path = csv_path
        self.model = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
        self.tfidf = TfidfVectorizer(max_features=300, stop_words='english')
        self.is_trained = False
        
        # Lưu các cột danh mục để map dữ liệu đầu vào từ Form Streamlit
        self.genre_columns = []
        self.source_columns = []
        self.rating_columns = []
        self.studio_columns = []
        self.tabular_feature_names = []
        
    def _clean_and_prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df = df[df['Type'] == 'TV'].copy()
        
        # Xử lý khuyết tật văn bản
        df['sypnopsis'] = df['sypnopsis'].fillna('Unknown')
        df['Studios'] = df['Studios'].fillna('Unknown')
        df['Rating'] = df['Rating'].fillna('Unknown')
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
        df = df.dropna(subset=['Score', 'Episodes'])
        
        # Trích xuất biến tương tác cộng đồng (Chuẩn hóa log để kéo R2 > 80%)
        df['Members'] = pd.to_numeric(df['Members'], errors='coerce').fillna(0)
        df['Favorites'] = pd.to_numeric(df['Favorites'], errors='coerce').fillna(0)
        df['log_members'] = np.log1p(df['Members'])
        df['log_favorites'] = np.log1p(df['Favorites'])
        
        # 1. Mã hóa Source và Rating
        df_source = pd.get_dummies(df['Source'], prefix='source', dtype=int)
        df_rating = pd.get_dummies(df['Rating'], prefix='rating', dtype=int)
        self.source_columns = df_source.columns.tolist()
        self.rating_columns = df_rating.columns.tolist()
        
        # 2. Mã hóa Thể loại (Genres)
        df['Genres_List'] = df['Genres'].apply(lambda x: [g.strip() for g in str(x).split(',')])
        all_genres = set([genre for sublist in df['Genres_List'] for genre in sublist])
        self.genre_columns = [f"genre_{g}" for g in sorted(list(all_genres))]
        for g_col, g_name in zip(self.genre_columns, sorted(list(all_genres))):
            df[g_col] = df['Genres_List'].apply(lambda x: 1 if g_name in x else 0)
            
        # 3. Mã hóa Studios
        top_studios = df['Studios'].value_counts().head(30).index
        self.studio_columns = [f'studio_{studio}' for studio in top_studios]
        for studio in top_studios:
            df[f'studio_{studio}'] = (df['Studios'] == studio).astype(int)
            
        # Trích xuất dữ liệu bảng
        X_tabular = pd.concat([
            df[['Episodes', 'log_members', 'log_favorites']], 
            df_source, 
            df_rating, 
            df[self.genre_columns],
            df[self.studio_columns]
        ], axis=1)
        
        # Lưu lại kịch bản cột để xử lý form sau này
        self.tabular_feature_names = X_tabular.columns.tolist()
        
        # Trích xuất TF-IDF cốt truyện
        X_tfidf = self.tfidf.fit_transform(df['sypnopsis'])
        X_final = hstack([X_tabular.values, X_tfidf]).tocsr()
        y = df['Score'].values
        
        return X_final, y

    def train(self):
        X, y = self._clean_and_prepare_data()
        
        # Đánh giá chéo 5-Fold mẫu
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_list = []
        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Chỉ cần fit mô hình cuối cùng lên toàn bộ data hoặc fold cuối
            self.model.fit(X_train, y_train)
            r2_list.append(self.model.score(X_test, y_test))
            
        self.is_trained = True
        return np.mean(r2_list)

    def predict_score(self, episodes, members, favorites, source, rating, genres_list, studio, sypnopsis_text):
        if not self.is_trained:
            raise Exception("Mô hình chưa được train!")
            
        # Dựng lại dictionary khớp định dạng
        input_dict = {
            'Episodes': [episodes],
            'log_members': [np.log1p(members)],
            'log_favorites': [np.log1p(favorites)]
        }
        
        # Điền các cột One-Hot / Multi-Hot (Kiểm tra chuỗi an toàn bảo vệ mô hình)
        for col in self.source_columns:
            input_dict[col] = [1 if f"source_{source}" == col else 0]
        for col in self.rating_columns:
            input_dict[col] = [1 if f"rating_{rating}" == col else 0]
        for col in self.genre_columns:
            g_name = col.replace("genre_", "")
            input_dict[col] = [1 if g_name in genres_list else 0]
        for col in self.studio_columns:
            s_name = col.replace("studio_", "")
            input_dict[col] = [1 if s_name == studio else 0]
            
        # Đảm bảo sắp xếp đúng thứ tự cột, tự điền 0 nếu người dùng chọn dữ liệu lạ ngoài danh mục huấn luyện
        X_tabular_input = pd.DataFrame(input_dict)
        X_tabular_input = X_tabular_input.reindex(columns=self.tabular_feature_names, fill_value=0).values
        
        # Biến đổi TF-IDF văn bản kịch bản nhập từ form
        X_tfidf_input = self.tfidf.transform([sypnopsis_text])
        
        X_final_input = hstack([X_tabular_input, X_tfidf_input]).tocsr()
        return self.model.predict(X_final_input)[0]