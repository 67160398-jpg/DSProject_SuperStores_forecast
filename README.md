# SuperStore Sales Forecast

## Overview
โปรเจคนี้วิเคราะห์ข้อมูลยอดขายของ SuperStore และสร้างแบบจำลองพยากรณ์ด้วยวิธี Time series โดยใช้ Auto ARIMA เพื่อทำนายยอดขายในอนาคต

## Objectives
- วิเคราะห์แนวโน้มยอดขาย
- พยากรณ์ยอดขายในอนาคต

## Tools & Technologies
- Python
- Pandas
- Statsmodels / pmdarima
- Streamlit

## Methodology
1. Data Cleaning
2. Time Series Analysis
3. Stationarity Test (ADF)
4. ARIMA / Auto ARIMA
5. Model Evaluation (MAE, RMSE)
6. Forecasting

## Result
- Auto ARIMA ได้ผลลัพธ์ที่ดีกว่า ได้ค่า MAE, RMSE น้อยกว่า ARIMA หลายเท่า ดังนั้นเราจะใช้ Model Auto ARIMA ไปใช้พยากรณ์
- แสดงกราฟพยากรณ์ของยอดการขายในอนาคต (1-2ปี)

## Deployment
Streamlit App: (https://dsprojectsuperstoresforecast-95r3hqo3evvpbzoysrzgit.streamlit.app/)

## Dataset
- SuperStoreOrders - SuperStoreOrders.csv
- url: https://www.kaggle.com/datasets/thuandao/superstore-sales-analytics
