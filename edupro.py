import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="EduPro Analytics", layout="wide")

st.title("🎓 EduPro - Course Demand & Revenue Predictor")

# 1. Load & Merge Data
@st.cache_data
def load_data():
    file_path = "EduPro Online Platform.xlsx"
    courses_df = pd.read_excel(file_path, sheet_name="Courses")
    teachers_df = pd.read_excel(file_path, sheet_name="Teachers")
    transactions_df = pd.read_excel(file_path, sheet_name="Transactions")

    course_stats = transactions_df.groupby('CourseID').agg(
        Total_Enrollments=('TransactionID', 'count'),
        Total_Revenue=('Amount', 'sum')
    ).reset_index()

    merged_df = courses_df.merge(course_stats, on='CourseID', how='left')
    merged_df = pd.concat([merged_df, teachers_df], axis=1)
    return merged_df

df = load_data()

st.subheader("📊 Dataset Overview")
st.dataframe(df[['CourseID', 'CourseName', 'CourseCategory', 'CoursePrice', 'Total_Enrollments', 'Total_Revenue']].head(10))

# 2. Preprocessing
df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=['CourseCategory', 'CourseType', 'CourseLevel'], drop_first=True)

features = [c for c in df_model.columns if c not in ['CourseID', 'CourseName', 'TeacherID', 'TeacherName', 'Email', 'Gender', 'Expertise', 'Total_Enrollments', 'Total_Revenue']]
X = df_model[features].fillna(0)
y_enroll = df_model['Total_Enrollments'].fillna(0)
y_rev = df_model['Total_Revenue'].fillna(0)

# 3. Train Models
model_enroll = RandomForestRegressor(random_state=42).fit(X, y_enroll)
model_rev = RandomForestRegressor(random_state=42).fit(X, y_rev)

st.markdown("---")
st.sidebar.header("⚙️ Set Course Features")

# Sidebar inputs
price = st.sidebar.slider("Course Price ($)", 10, 500, 100)
duration = st.sidebar.slider("Course Duration (Hours)", 1, 100, 20)
rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.2)
exp = st.sidebar.slider("Teacher Experience (Years)", 1, 20, 5)

btn = st.sidebar.button("Predict Performance")

# Main Screen Prediction Section
st.subheader("🎯 Prediction Output")

if btn:
    input_data = X.iloc[0:1].copy()
    if 'CoursePrice' in input_data.columns: input_data['CoursePrice'] = price
    if 'CourseDuration' in input_data.columns: input_data['CourseDuration'] = duration
    if 'CourseRating' in input_data.columns: input_data['CourseRating'] = rating
    if 'YearsOfExperience' in input_data.columns: input_data['YearsOfExperience'] = exp

    pred_e = int(model_enroll.predict(input_data)[0])
    pred_r = round(float(model_rev.predict(input_data)[0]), 2)

    col1, col2 = st.columns(2)
    col1.metric("Predicted Enrollments", f"{pred_e} Users")
    col2.metric("Predicted Revenue", f"${pred_r}")
    st.success("Prediction generated successfully!")
else:
    st.info("Sidebar me sliders set karke 'Predict Performance' button par click karein.")