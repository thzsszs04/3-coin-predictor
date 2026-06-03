from __future__ import annotations

import html
import json
import locale
import os
import re
import ssl
from datetime import datetime
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import certifi
except ImportError:
    certifi = None

from src.predictor import COIN_OPTIONS, DEFAULT_DATA_PATH, MODEL_OPTIONS, load_dataset, run_backtesting, run_prediction


APP_TITLE = "3-Coin Predictor"
TIMEZONE = ZoneInfo("Asia/Jakarta")
STYLE_PATH = Path(__file__).with_name("style.css")
AI_INSIGHT_CACHE_VERSION = "model-limitations-v11"

TEXT = {
    "id": {
        "nav_prediction": "Prediksi Kripto",
        "nav_comparison": "Komparasi Performa<br>Model",
        "nav_variable": "Variabel",
        "nav_dataset": "Dataset",
        "nav_backtesting": "Backtesting",
        "nav_method": "Metode",
        "comparison_title": "Perbandingan Performa Model",
        "comparison_subtitle": "Komparasi performa model SARIMAX dan XGBoost.",
        "comparison_start": "Lihat Hasil Perbandingan",
        "comparison_loading": "Memuat perbandingan model...",
        "comparison_warning": "Silakan pilih koin terlebih dahulu sebelum melihat hasil perbandingan.",
        "comparison_best_title": "Koin dan Model dengan Performa Terbaik",
        "comparison_best_coin": "Koin",
        "comparison_best_model": "Model",
        "comparison_table_title": "Tabel Perbandingan Performa SARIMAX VS XGBoost",
        "comparison_table_placeholder": "Tabel perbandingan metrik evaluasi untuk SARIMAX dan XGBoost akan muncul setelah perbandingan dijalankan.",
        "comparison_error_title": "Visualisasi Komparasi Metrik Error Model SARIMAX vs XGBoost",
        "comparison_error_placeholder": "Grouped bar chart visualisasi komparasi metrik error akan muncul setelah perbandingan dijalankan.",
        "comparison_direction_title": "Visualisasi Komparasi Directional Accuracy Model SARIMAX vs XGBoost",
        "comparison_direction_placeholder": "Bar chart visualisasi perbandingan Directional Accuracy akan muncul setelah perbandingan dijalankan.",
        "comparison_return_title": "Visualisasi Komparasi Cumulative Return Model SARIMAX vs XGBoost",
        "comparison_return_placeholder": "Line chart visualisasi perbandingan Cumulative Return akan muncul setelah perbandingan dijalankan.",
        "comparison_ai_placeholder": "Interpretasi hasil perbandingan model akan muncul setelah perbandingan dijalankan.",
        "comparison_ai_loading": "Menganalisis perbandingan model...",
        "method_title": "Metode",
        "method_subtitle": "Penjelasan model prediksi yang digunakan dalam penelitian.",
        "method_intro_title": "Ringkasan Pendek",
        "method_intro_body": "Website ini dikembangkan untuk mendukung riset pengembang dalam membandingkan performa model statistik dan model machine learning untuk memprediksi pasar kripto. Karena itu, website ini menggunakan dua pendekatan yang saling melengkapi: SARIMAX sebagai model statistik untuk membaca pola harga berdasarkan urutan waktu, dan XGBoost sebagai model machine learning untuk mempelajari hubungan non-linear dari harga kripto serta indikator eksternal seperti volume, volatilitas, dan Fear & Greed.",
        "backtesting_title": "Simulasi Backtesting",
        "backtesting_subtitle": "Menguji seberapa baik model yang dipilih memprediksi harga kripto selama periode historis yang dipilih.",
        "choose_date": "Pilih Tanggal",
        "reset_date": "Reset",
        "date_options": "Tanggal awal simulasi historis",
        "start_backtest": "Mulai Backtest",
        "loading_backtest": "Memuat backtesting...",
        "backtest_select_warning": "Silakan pilih koin, model, dan tanggal terlebih dahulu sebelum memulai backtest.",
        "backtest_chart_title": "Grafik Hasil Backtesting",
        "backtest_table_title": "Tabel Hasil Backtesting",
        "backtest_chart_placeholder": "Visualisasi chart harga aktual vs prediksi dari backtesting akan muncul setelah backtest dijalankan.",
        "backtest_table_placeholder": "Tabel hasil backtesting berisi tanggal, harga aktual, harga prediksi, error, dan trend.",
        "prediction_reliability": "Reliabilitas Prediksi",
        "confidence_level": "Confidence Level",
        "backtest_period": "Periode Backtesting",
        "backtest_ai_placeholder": "Interpretasi hasil backtesting dan kondisi pasar pada periode tersebut akan muncul setelah backtest dijalankan.",
        "backtest_ai_loading": "Menganalisis hasil backtesting...",
        "backtest_ai_refs": "Referensi berita lanjutan",
        "prediction_title": "Prediksi Kripto",
        "prediction_subtitle": "Hasil prediksi model untuk data testing.",
        "live_market_title": "Live Market Overview",
        "live_price": "Harga Live",
        "change_24h": "Perubahan 24 Jam",
        "market_signal": "Sinyal Market",
        "fear_greed_title": "Fear & Greed Index",
        "fear_greed_source": "Sumber live sentimen pasar kripto",
        "live_unavailable": "Data live belum dapat dimuat.",
        "live_status_ready": "Menunggu update live",
        "live_status_success": "Live diperbarui",
        "live_status_failed": "API live gagal, memakai data terakhir",
        "choose_coin": "Pilih Koin",
        "choose_model": "Pilih Model",
        "choose_horizon": "Pilih Horizon",
        "horizon_placeholder": "Pilih Horizon",
        "coin_placeholder": "Pilih Koin",
        "model_placeholder": "Pilih Model",
        "horizon_options": "Opsi: 1D, 3D, 7D",
        "coin_options": "Opsi: Bitcoin, Ethereum, Solana",
        "model_options": "Opsi: SARIMAX, XGBoost",
        "action": "Aksi",
        "start_prediction": "Mulai Prediksi",
        "select_warning": "Silakan pilih koin, model, dan horizon terlebih dahulu sebelum memulai prediksi.",
        "loading_prediction": "Memuat prediksi...",
        "actual_vs_prediction": "Harga Aktual vs Prediksi",
        "actual_price": "Harga Aktual",
        "predicted_price": "Harga Prediksi",
        "historical_price": "Harga Historis",
        "seven_day_prediction": "Prediksi Forecast",
        "chart_placeholder": "Pilih koin dan model, lalu klik Mulai Prediksi untuk menampilkan chart harga aktual vs prediksi.",
        "chart_caption": "Visualisasi chart harga aktual vs prediksi berdasarkan data testing.",
        "test_period": "Periode testing",
        "metrics_title": "Metrik Kinerja Prediksi",
        "seven_day_title": "Prediksi Kripto {days} Hari Ke Depan",
        "forecast_placeholder": "Forecast akan muncul setelah model dijalankan.",
        "ai_placeholder": "Interpretasi grafik forecast beserta penjelasan terkait sentimen global akan muncul setelah prediksi dijalankan.",
        "feature_title": "Diagram Bar Feature Importance Model XGBoost",
        "feature_caption": "Diagram ini menampilkan 10 variabel teratas yang paling berpengaruh terhadap keputusan model XGBoost dalam memprediksi harga kripto terpilih.",
        "sarimax_diagnostics_title": "Diagnostik Residual Model SARIMAX",
        "residual_distribution_title": "Distribusi Residual",
        "residual_autocorrelation_title": "Autocorrelation Plot Residual",
        "ljung_box_title": "Hasil Ljung-Box Test",
        "ljung_box_lag": "Lag",
        "ljung_box_stat": "Statistik",
        "ljung_box_pvalue": "P-Value",
        "ljung_box_result": "Interpretasi",
        "ljung_box_pass": "Residual relatif acak",
        "ljung_box_pattern": "Masih ada pola residual",
        "residual_conclusion": "Kesimpulan",
        "residual_conclusion_good": "Model cukup baik menangkap pola residual",
        "residual_conclusion_watch": "Pola residual masih perlu diperhatikan",
        "residual_conclusion_helper_good": "Tidak ada pola residual kuat yang terdeteksi",
        "residual_conclusion_helper_watch": "Masih ada indikasi pola tersisa pada residual",
        "ljung_box_tooltip": "Ljung-Box Test memeriksa apakah residual masih memiliki pola berulang. P-Value di atas 0.05 biasanya menunjukkan residual relatif acak.",
        "residual_conclusion_tooltip": "Kesimpulan ringkas dari hasil Ljung-Box. Jika residual relatif acak, berarti pola utama data sudah cukup baik ditangkap oleh SARIMAX.",
        "sarimax_diagnostics_caption": "Diagnostik ini membantu memeriksa apakah residual SARIMAX sudah cukup acak. Residual yang acak menunjukkan pola utama data sudah lebih banyak ditangkap oleh model.",
        "local_data_warning": "Data live gagal dimuat, sehingga dashboard memakai CSV lokal. Periode testing bisa tidak mengikuti rolling 2 tahun terbaru.",
        "ai_loading": "Menganalisis forecast dan sentimen pasar...",
        "ai_missing": "API AI belum dikonfigurasi. Tambahkan kredensial API ke <strong>.streamlit/secrets.toml</strong> atau environment variable untuk mengaktifkan analisis forecast, sentimen kripto, dan konteks berita global.",
        "ai_refs": "Referensi berita lanjutan",
        "page_loading": "Memuat halaman...",
        "dataset_title": "Dataset",
        "dataset_subtitle": "Ringkasan dan preview dataset final yang digunakan dalam penelitian.",
        "dataset_loading": "Memuat dataset...",
        "dataset_summary": "Ringkasan Dataset",
        "data_source": "Sumber Data",
        "row_count": "Jumlah Baris",
        "column_count": "Jumlah Kolom",
        "date_range": "Rentang Tanggal",
        "active_dataset_source": "Sumber dataset aktif",
        "total_rows": "Total observasi setelah preprocessing",
        "total_columns": "Total variabel dataset final",
        "final_period": "Periode data final",
        "live_source": "Live rolling 2 tahun",
        "local_source": "CSV lokal",
        "dataset_preview": "Preview Dataset",
        "dataset_caption": "Tabel berikut menampilkan seluruh baris dari versi final dataset setelah preprocessing. Gunakan scroll horizontal dan vertikal untuk melihat semua variabel.",
        "variable_title": "Variabel",
        "variable_subtitle": "Indikator market eksternal dan variabel data yang digunakan dalam model prediksi.",
        "variable_loading": "Memuat variabel...",
        "variable_goal": "Tujuan:",
        "variable_intro": "Halaman ini memberikan transparansi mengenai variabel dan indikator pasar eksternal yang digunakan dalam model prediksi harga mata uang kripto (SARIMAX dan XGBoost). Memahami variabel-variabel ini membantu menafsirkan prediksi model dan menilai faktor-faktor yang memengaruhi pergerakan harga mata uang kripto.",
        "description": "DESKRIPSI:",
        "data_source_label": "SUMBER DATA:",
        "prediction_relevance": "RELEVANSI DENGAN PREDIKSI HARGA:",
        "final_variables": "Daftar Variabel Final",
        "final_variables_caption": "Tabel ini merangkum seluruh variabel pada versi final dataset setelah preprocessing yang digunakan dalam penelitian.",
        "placeholder_subtitle": "Halaman ini akan dibuat pada tahap berikutnya.",
        "placeholder_body": "{title} akan dibuat pada tahap berikutnya.",
        "metric_directional": "Ketepatan Arah Prediksi",
        "metric_return": "Imbal Hasil Kumulatif",
        "var_column": "Variabel",
        "category_column": "Kategori",
        "desc_column": "Deskripsi",
        "role_column": "Peran",
    },
    "en": {
        "nav_prediction": "Crypto Prediction",
        "nav_comparison": "Model Performance<br>Comparison",
        "nav_variable": "Variables",
        "nav_dataset": "Dataset",
        "nav_backtesting": "Backtesting",
        "nav_method": "Method",
        "comparison_title": "Model Performance Comparison",
        "comparison_subtitle": "Performance comparison between SARIMAX and XGBoost models.",
        "comparison_start": "View Comparison Result",
        "comparison_loading": "Loading model comparison...",
        "comparison_warning": "Please choose a coin before viewing the comparison result.",
        "comparison_best_title": "Coin and Best Performing Model",
        "comparison_best_coin": "Coin",
        "comparison_best_model": "Model",
        "comparison_table_title": "SARIMAX VS XGBoost Performance Comparison Table",
        "comparison_table_placeholder": "The evaluation metric comparison table for SARIMAX and XGBoost will appear after the comparison is run.",
        "comparison_error_title": "SARIMAX vs XGBoost Error Metric Comparison Visualization",
        "comparison_error_placeholder": "The grouped bar chart comparing error metrics will appear after the comparison is run.",
        "comparison_direction_title": "SARIMAX vs XGBoost Directional Accuracy Comparison Visualization",
        "comparison_direction_placeholder": "The Directional Accuracy comparison bar chart will appear after the comparison is run.",
        "comparison_return_title": "SARIMAX vs XGBoost Cumulative Return Comparison Visualization",
        "comparison_return_placeholder": "The Cumulative Return comparison line chart will appear after the comparison is run.",
        "comparison_ai_placeholder": "The model comparison interpretation will appear after the comparison is run.",
        "comparison_ai_loading": "Analyzing model comparison...",
        "method_title": "Method",
        "method_subtitle": "Explanation of the prediction models used in this research.",
        "method_intro_title": "Short Summary",
        "method_intro_body": "This website was developed to support the developer's research in comparing the performance of a statistical model and a machine learning model for predicting the crypto market. For that reason, this website uses two complementary approaches: SARIMAX as the statistical model for reading price patterns over time, and XGBoost as the machine learning model for learning non-linear relationships from crypto prices and external indicators such as volume, volatility, and Fear & Greed.",
        "backtesting_title": "Backtesting Simulation",
        "backtesting_subtitle": "Test how well the selected model predicted crypto prices over the selected historical period.",
        "choose_date": "Choose Date",
        "reset_date": "Reset",
        "date_options": "Historical simulation start date",
        "start_backtest": "Start Backtest",
        "loading_backtest": "Loading backtest...",
        "backtest_select_warning": "Please choose a coin, model, and date before starting the backtest.",
        "backtest_chart_title": "Backtesting Result Chart",
        "backtest_table_title": "Backtesting Result Table",
        "backtest_chart_placeholder": "The actual vs predicted backtesting chart will appear after the backtest is run.",
        "backtest_table_placeholder": "The backtesting table contains date, actual price, predicted price, error, and trend.",
        "prediction_reliability": "Prediction Reliability",
        "confidence_level": "Confidence Level",
        "backtest_period": "Backtesting Period",
        "backtest_ai_placeholder": "The interpretation of backtesting results and market conditions during that period will appear after the backtest is run.",
        "backtest_ai_loading": "Analyzing backtesting result...",
        "backtest_ai_refs": "Further news references",
        "prediction_title": "Crypto Prediction",
        "prediction_subtitle": "Model prediction results for the testing data.",
        "live_market_title": "Live Market Overview",
        "live_price": "Live Price",
        "change_24h": "24H Change",
        "market_signal": "Market Signal",
        "fear_greed_title": "Fear & Greed Index",
        "fear_greed_source": "Live crypto market sentiment source",
        "live_unavailable": "Live data could not be loaded yet.",
        "live_status_ready": "Waiting for live update",
        "live_status_success": "Live updated",
        "live_status_failed": "Live API failed, using last data",
        "choose_coin": "Choose Coin",
        "choose_model": "Choose Model",
        "choose_horizon": "Choose Horizon",
        "horizon_placeholder": "Choose Horizon",
        "coin_placeholder": "Choose Coin",
        "model_placeholder": "Choose Model",
        "horizon_options": "Options: 1D, 3D, 7D",
        "coin_options": "Options: Bitcoin, Ethereum, Solana",
        "model_options": "Options: SARIMAX, XGBoost",
        "action": "Action",
        "start_prediction": "Start Prediction",
        "select_warning": "Please choose a coin, model, and horizon before starting the prediction.",
        "loading_prediction": "Loading prediction...",
        "actual_vs_prediction": "Actual Price vs Prediction",
        "actual_price": "Actual Price",
        "predicted_price": "Predicted Price",
        "historical_price": "Historical Price",
        "seven_day_prediction": "Forecast Prediction",
        "chart_placeholder": "Choose a coin and model, then click Start Prediction to show the actual vs predicted price chart.",
        "chart_caption": "Visualization of actual vs predicted prices based on testing data.",
        "test_period": "Testing period",
        "metrics_title": "Prediction Performance Metrics",
        "seven_day_title": "Crypto Prediction for the Next {days} Days",
        "forecast_placeholder": "The forecast will appear after the model is run.",
        "ai_placeholder": "The interpretation of the forecast chart and global sentiment explanation will appear after the prediction is run.",
        "feature_title": "XGBoost Model Feature Importance Bar Chart",
        "feature_caption": "This chart shows the top 10 variables that most influence the XGBoost model when predicting the selected crypto price.",
        "sarimax_diagnostics_title": "SARIMAX Model Residual Diagnostics",
        "residual_distribution_title": "Residual Distribution",
        "residual_autocorrelation_title": "Residual Autocorrelation Plot",
        "ljung_box_title": "Ljung-Box Test Result",
        "ljung_box_lag": "Lag",
        "ljung_box_stat": "Statistic",
        "ljung_box_pvalue": "P-Value",
        "ljung_box_result": "Interpretation",
        "ljung_box_pass": "Residuals are relatively random",
        "ljung_box_pattern": "Residual pattern remains",
        "residual_conclusion": "Conclusion",
        "residual_conclusion_good": "The model captures the residual pattern fairly well",
        "residual_conclusion_watch": "Residual patterns still need attention",
        "residual_conclusion_helper_good": "No strong residual pattern is detected",
        "residual_conclusion_helper_watch": "Remaining residual pattern is still indicated",
        "ljung_box_tooltip": "The Ljung-Box Test checks whether residuals still contain repeated patterns. A P-Value above 0.05 usually suggests the residuals are relatively random.",
        "residual_conclusion_tooltip": "A simple conclusion from the Ljung-Box result. If residuals are relatively random, SARIMAX has captured the main data pattern fairly well.",
        "sarimax_diagnostics_caption": "These diagnostics help check whether SARIMAX residuals are sufficiently random. Random residuals suggest the model has captured more of the main data pattern.",
        "local_data_warning": "Live data failed to load, so the dashboard is using the local CSV. The testing period may not follow the latest rolling 2-year window.",
        "ai_loading": "Analyzing forecast and market sentiment...",
        "ai_missing": "The AI API is not configured yet. Add API credentials to <strong>.streamlit/secrets.toml</strong> or an environment variable to enable forecast, crypto sentiment, and global context analysis.",
        "ai_refs": "Further news references",
        "page_loading": "Loading page...",
        "dataset_title": "Dataset",
        "dataset_subtitle": "Summary and preview of the final dataset used in this research.",
        "dataset_loading": "Loading dataset...",
        "dataset_summary": "Dataset Summary",
        "data_source": "Data Source",
        "row_count": "Rows",
        "column_count": "Columns",
        "date_range": "Date Range",
        "active_dataset_source": "Active dataset source",
        "total_rows": "Total observations after preprocessing",
        "total_columns": "Total final dataset variables",
        "final_period": "Final data period",
        "live_source": "Live rolling 2 years",
        "local_source": "Local CSV",
        "dataset_preview": "Dataset Preview",
        "dataset_caption": "The table below displays all rows from the final version of the dataset after preprocessing. Use horizontal and vertical scrolling to inspect all variables.",
        "variable_title": "Variables",
        "variable_subtitle": "External market indicators and data variables used in the prediction models.",
        "variable_loading": "Loading variables...",
        "variable_goal": "Purpose:",
        "variable_intro": "This page provides transparency about the variables and external market indicators used in the cryptocurrency price prediction models (SARIMAX and XGBoost). Understanding these variables helps interpret the model predictions and assess factors that may influence crypto price movements.",
        "description": "DESCRIPTION:",
        "data_source_label": "DATA SOURCE:",
        "prediction_relevance": "RELEVANCE TO PRICE PREDICTION:",
        "final_variables": "Final Variable List",
        "final_variables_caption": "This table summarizes all variables in the final version of the dataset after preprocessing used in the research.",
        "placeholder_subtitle": "This page will be built in the next stage.",
        "placeholder_body": "{title} will be built in the next stage.",
        "metric_directional": "Prediction Direction Accuracy",
        "metric_return": "Cumulative Return",
        "var_column": "Variable",
        "category_column": "Category",
        "desc_column": "Description",
        "role_column": "Role",
    },
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> str:
    base_css = STYLE_PATH.read_text(encoding="utf-8") if STYLE_PATH.exists() else ""
    overrides = """
    <style>
        :root {
            --sidebar-bg: #223044;
            --panel-bg: #263545;
            --active-bg: #314257;
            --hover-bg: #263545;
            --button-bg: #38D9A9;
            --button-hover: #4DB8FF;
            --text-main: #F8FAFC;
            --text-muted: #C0CAD8;
            --border: #425168;
        }

        .stApp {
            background: #182231;
            color: var(--text-main);
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stDeployButton,
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        section[data-testid="stSidebar"] {
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            bottom: 0 !important;
            transform: none !important;
            visibility: visible !important;
        }

        [data-testid="stAppViewContainer"] {
            margin-left: 280px;
            width: calc(100% - 280px);
        }

        [data-testid="stMain"] {
            padding-top: 0 !important;
        }

        [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }

        .main .block-container {
            padding: 0.1rem 3.4rem 5.4rem 3.4rem;
            max-width: 100%;
        }

        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            width: 280px !important;
            min-width: 280px !important;
            padding-top: 0.25rem;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 0.9rem 1.35rem 1.5rem 1.35rem;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] p {
            color: var(--text-main);
        }

        .brand-logo {
            position: relative;
            display: grid;
            grid-template-columns: 54px minmax(0, 1fr);
            gap: 0.8rem;
            align-items: center;
            padding: 0.85rem 0.75rem 0.95rem 0.75rem;
            margin-bottom: 1.15rem;
            border: 1px solid #3A4B61;
            border-radius: 14px;
            background:
                radial-gradient(circle at 18% 18%, rgba(56, 217, 169, 0.2), transparent 38%),
                linear-gradient(135deg, rgba(49, 66, 87, 0.98), rgba(38, 53, 69, 0.84));
            box-shadow: 0 12px 28px rgba(5, 10, 18, 0.22);
            overflow: hidden;
        }

        .brand-logo::before {
            content: "";
            position: absolute;
            top: -45%;
            bottom: -45%;
            left: -75%;
            width: 42%;
            z-index: 1;
            background: linear-gradient(
                100deg,
                transparent 0%,
                rgba(255, 255, 255, 0.08) 28%,
                rgba(255, 255, 255, 0.42) 50%,
                rgba(255, 255, 255, 0.08) 72%,
                transparent 100%
            );
            transform: skewX(-18deg);
            animation: brandWhiteSweep 3s ease-in-out infinite;
        }

        .brand-logo > * {
            position: relative;
            z-index: 2;
        }

        .brand-mark {
            position: relative;
            width: 48px;
            height: 48px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            background:
                linear-gradient(135deg, rgba(56, 217, 169, 0.22), rgba(77, 184, 255, 0.18)),
                #182231;
            border: 1px solid rgba(56, 217, 169, 0.55);
            overflow: hidden;
            box-shadow: inset 0 0 18px rgba(77, 184, 255, 0.08), 0 0 18px rgba(56, 217, 169, 0.14);
        }

        .brand-mark::before {
            content: "";
            position: absolute;
            width: 74px;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(56, 217, 169, 0.9), transparent);
            transform: rotate(-32deg);
        }

        .brand-core {
            position: relative;
            z-index: 2;
            font-family: "Courier New", monospace;
            font-size: 1.85rem;
            line-height: 1;
            font-weight: 900;
            color: #F8FAFC;
            text-shadow: 0 0 12px rgba(56, 217, 169, 0.62);
            transform-style: preserve-3d;
            animation: brandCoreRotate 6s linear infinite;
        }

        .brand-orbit {
            position: absolute;
            border: 1px solid rgba(77, 184, 255, 0.45);
            border-radius: 999px;
            transform: rotate(-26deg);
        }

        .orbit-one {
            width: 40px;
            height: 20px;
        }

        .orbit-two {
            width: 22px;
            height: 38px;
            border-color: rgba(56, 217, 169, 0.42);
            transform: rotate(32deg);
        }

        .brand-name {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 0.18rem;
            color: #F8FAFC;
            font-size: 1.14rem;
            line-height: 1.05;
            letter-spacing: 0.01em;
            font-weight: 800;
        }

        .brand-name span {
            background: linear-gradient(135deg, #38D9A9, #4DB8FF);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .brand-name strong {
            color: #F8FAFC;
            font-weight: 900;
        }

        @keyframes brandWhiteSweep {
            0%, 28% {
                left: -75%;
                opacity: 0;
            }
            38% {
                opacity: 1;
            }
            62% {
                opacity: 0.72;
            }
            78%, 100% {
                left: 130%;
                opacity: 0;
            }
        }

        @keyframes brandCoreRotate {
            0% {
                transform: rotateY(0deg);
            }
            100% {
                transform: rotateY(360deg);
            }
        }

        .sidebar-title {
            font-size: 1.65rem;
            line-height: 1.2;
            font-weight: 800;
            margin: 0 0 1.25rem 0;
            color: var(--text-main);
        }

        .sidebar-subtitle {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1.35rem;
            color: var(--text-main);
        }

        .sidebar-divider {
            height: 2px;
            background: var(--border);
            margin: 0 0 1.8rem 0;
        }

        .nav-link {
            position: relative;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-height: 58px;
            padding: 0.75rem 1rem;
            border-radius: 5px;
            color: var(--text-muted);
            text-decoration: none !important;
            border: 1px solid #3A4B61 !important;
            border-left: 4px solid transparent !important;
            font-weight: 700;
            margin-bottom: 0.85rem;
            background: #223044;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
            overflow: hidden;
        }

        .nav-link::after {
            content: "";
            position: absolute;
            top: -45%;
            bottom: -45%;
            left: -70%;
            width: 38%;
            z-index: 1;
            pointer-events: none;
            opacity: 0;
            background: linear-gradient(
                100deg,
                transparent 0%,
                rgba(255, 255, 255, 0.06) 28%,
                rgba(255, 255, 255, 0.34) 50%,
                rgba(255, 255, 255, 0.06) 72%,
                transparent 100%
            );
            transform: skewX(-18deg);
        }

        .nav-link:hover::after {
            animation: menuWhiteSweep 0.72s ease-out 1;
        }

        .nav-link *,
        .nav-link:visited,
        .nav-link:hover,
        .nav-link:active {
            text-decoration: none !important;
        }

        .nav-link:hover {
            background-color: var(--hover-bg);
            border-color: #4E637D !important;
            border-left-color: #38D9A9 !important;
            color: #38D9A9 !important;
        }

        .nav-link.active {
            background-color: var(--active-bg);
            border-color: #58708C !important;
            border-left-color: #38D9A9 !important;
            color: var(--text-main) !important;
            animation: activeMenuPulse 2.5s ease-in-out infinite;
        }

        .nav-link > * {
            position: relative;
            z-index: 2;
        }

        @keyframes menuWhiteSweep {
            0% {
                left: -70%;
                opacity: 0;
            }
            22% {
                opacity: 0.95;
            }
            100% {
                left: 135%;
                opacity: 0;
            }
        }

        @keyframes activeMenuPulse {
            0%, 100% {
                border-color: #58708C;
                box-shadow: 0 0 0 rgba(56, 217, 169, 0), inset 0 0 0 rgba(56, 217, 169, 0);
            }
            50% {
                border-color: rgba(56, 217, 169, 0.86);
                box-shadow: 0 0 22px rgba(56, 217, 169, 0.34), inset 0 0 22px rgba(56, 217, 169, 0.11);
            }
        }

        .nav-icon {
            width: 1.8rem;
            font-size: 1.25rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 0;
        }

        .nav-text {
            line-height: 1.15;
        }

        .page-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: start;
            gap: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
            margin-bottom: 1.2rem;
        }

        .page-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.45rem 0;
        }

        .page-subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            font-weight: 500;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding-top: 2.2rem;
        }

        .language-switch {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--border);
            border-radius: 999px;
            overflow: hidden;
            background: #182231;
        }

        .language-switch a {
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.78rem;
            font-weight: 850;
            padding: 0.35rem 0.7rem;
            line-height: 1;
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        .language-switch a:hover {
            color: #38D9A9;
            background: #263545;
        }

        .language-switch a.active {
            color: #182231;
            background: #38D9A9;
        }

        .language-switch a.active:hover {
            color: #182231;
            background: #4DB8FF;
        }

        .date-label {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-muted);
            white-space: nowrap;
        }

        .control-panel {
            background: var(--panel-bg);
            padding: 1.35rem 1.45rem;
            margin-bottom: 2.6rem;
        }

        .st-key-control_panel {
            background: var(--panel-bg);
            border-radius: 10px;
            padding: 1.35rem 1.45rem 1.05rem 1.45rem;
            margin-bottom: 2.35rem;
        }

        .st-key-control_panel [data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        .control-help {
            color: var(--text-muted);
            font-size: 0.76rem;
            margin-top: -0.4rem;
        }

        .button-label-spacer {
            color: transparent;
            font-size: 0.98rem;
            font-weight: 600;
            line-height: 1.5;
            margin-bottom: 0.5rem;
            user-select: none;
        }

        div[data-testid="stSelectbox"] label {
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.98rem;
        }

        div[data-testid="stDateInput"] label {
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.98rem;
        }

        div[data-baseweb="select"] > div {
            background-color: #182231;
            border: 1px solid var(--border);
            border-radius: 8px;
            min-height: 54px;
            color: var(--text-main);
            box-shadow: none;
        }

        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
        }

        div[data-testid="stDateInput"] input {
            background-color: #182231 !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            min-height: 54px !important;
            color: var(--text-main) !important;
            box-shadow: none !important;
            padding-left: 0.9rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stDateInput"] input::placeholder {
            color: var(--text-muted) !important;
            opacity: 1 !important;
        }

        div.stButton > button:first-child {
            background: linear-gradient(135deg, #38D9A9 0%, #4DB8FF 100%);
            color: #182231;
            border: none;
            border-radius: 8px;
            width: 100%;
            min-height: 54px;
            margin-top: 0;
            font-weight: 800;
            transition: background-color 0.3s ease;
        }

        div.stButton > button:first-child:hover {
            color: #182231;
            border: none;
        }

        div.stFormSubmitButton > button:first-child {
            background: linear-gradient(135deg, #38D9A9 0%, #4DB8FF 100%);
            color: #182231;
            border: none;
            border-radius: 8px;
            width: 100%;
            min-height: 54px;
            margin-top: 0;
            font-weight: 800;
            transition: background-color 0.3s ease;
        }

        div.stFormSubmitButton > button:first-child:hover {
            color: #182231;
            border: none;
        }

        .section-title {
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0 0 1.35rem 0;
        }

        .chart-shell {
            background: var(--panel-bg);
            min-height: 360px;
            padding: 0.75rem;
            margin-bottom: 2.2rem;
        }

        .chart-caption {
            text-align: center;
            color: var(--text-muted);
            font-weight: 600;
            margin: 0.2rem 0 2rem 0;
        }

        .live-market-bottom-space {
            height: 1.35rem;
        }

        .metric-card {
            background: var(--panel-bg);
            border-radius: 5px;
            border: 1px solid var(--border);
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 126px;
            margin-bottom: 1rem;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.85rem;
        }

        .metric-value {
            color: var(--text-main);
            font-size: 2.25rem;
            font-weight: 500;
            line-height: 1;
            margin-bottom: 0.85rem;
        }

        .metric-help {
            color: var(--text-muted);
            font-size: 0.98rem;
            font-weight: 600;
        }

        .placeholder {
            background: var(--panel-bg);
            border-radius: 10px;
            padding: 2rem;
            color: var(--text-muted);
            font-weight: 600;
            text-align: center;
            margin-bottom: 2rem;
        }

        .stDataFrame {
            margin-top: 0.7rem;
        }

        @media (max-width: 900px) {
            [data-testid="stAppViewContainer"] {
                margin-left: 280px;
                width: calc(100% - 280px);
            }

            .main .block-container {
                padding: 0.1rem 1.1rem 5.4rem 1.1rem;
            }

            .page-header {
                grid-template-columns: 1fr;
                gap: 0.4rem;
                margin-bottom: 1.8rem;
            }

            .header-actions {
                padding-top: 0;
                justify-content: space-between;
            }

            .date-label {
                padding-top: 0;
            }
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main .block-container {
            max-width: 100vw !important;
            overflow-x: hidden !important;
        }

        img,
        svg,
        canvas,
        iframe,
        .js-plotly-plot {
            max-width: 100% !important;
        }

        [data-testid="stHorizontalBlock"],
        [data-testid="stHorizontalBlock"] > div {
            min-width: 0 !important;
        }

        @media (max-width: 1366px) {
            .main .block-container {
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }

            .metric-card-grid,
            .backtest-summary-grid,
            .dataset-summary-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }

            .forecast-confidence-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }
        }

        @media (max-width: 980px) {
            section[data-testid="stSidebar"] {
                position: relative !important;
                left: auto !important;
                top: auto !important;
                bottom: auto !important;
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                height: auto !important;
                border-right: 0 !important;
                border-bottom: 1px solid #314257 !important;
            }

            [data-testid="stSidebar"] {
                width: 100% !important;
                min-width: 0 !important;
            }

            [data-testid="stSidebar"] > div:first-child {
                padding: 0.85rem 1rem 0.75rem 1rem !important;
            }

            [data-testid="stAppViewContainer"] {
                margin-left: 0 !important;
                width: 100% !important;
            }

            .app-footer {
                left: 0 !important;
                right: 0 !important;
                width: 100% !important;
            }

            .sidebar-divider {
                display: block !important;
                height: 1px !important;
                background: #425168 !important;
                margin: 0 0 1rem 0 !important;
            }

            .main .block-container {
                padding: 1rem 1rem 6.7rem 1rem !important;
            }

            .brand-logo {
                grid-template-columns: 44px minmax(0, 1fr) !important;
                margin-bottom: 0.85rem !important;
                padding: 0.65rem 0.7rem !important;
            }

            .brand-mark {
                width: 40px !important;
                height: 40px !important;
                border-radius: 12px !important;
            }

            .brand-core {
                font-size: 1.45rem !important;
            }

            .brand-name {
                font-size: 1rem !important;
            }

            .sidebar-subtitle,
            .sidebar-divider {
                display: none !important;
            }

            .nav-link {
                min-height: 48px !important;
                margin-bottom: 0.55rem !important;
                padding: 0.58rem 0.75rem !important;
                font-size: 0.9rem !important;
            }

            .nav-icon {
                width: 1.45rem !important;
                font-size: 1.05rem !important;
            }

            .page-header {
                grid-template-columns: 1fr !important;
                gap: 0.85rem !important;
                margin-bottom: 1.25rem !important;
            }

            .header-actions {
                padding-top: 0 !important;
                justify-content: space-between !important;
                flex-wrap: wrap !important;
            }

            .date-label {
                white-space: normal !important;
                text-align: right !important;
            }

            .app-footer {
                left: 0 !important;
                min-height: auto !important;
                padding: 0.7rem 0.9rem !important;
            }
        }

        @media (max-width: 820px) {
            [data-testid="stHorizontalBlock"],
            [data-testid="stHorizontalBlock"] > div {
                display: flex !important;
                flex-direction: column !important;
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            .st-key-control_panel [data-testid="stHorizontalBlock"],
            .st-key-control_panel [data-testid="stHorizontalBlock"] > div {
                display: flex !important;
                flex-direction: column !important;
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            .button-label-spacer {
                display: none !important;
            }

            .metric-card-grid,
            .dataset-summary-grid,
            .backtest-summary-grid,
            .forecast-confidence-grid,
            .method-page,
            .method-grid {
                grid-template-columns: 1fr !important;
            }

            .method-intro,
            .method-symbol-row {
                grid-template-columns: 1fr !important;
            }

            .page-title {
                font-size: clamp(1.65rem, 7vw, 2rem) !important;
            }

            .section-title {
                font-size: clamp(1.25rem, 5.4vw, 1.55rem) !important;
                line-height: 1.25 !important;
            }
        }

        @media (max-width: 640px) {
            .main .block-container {
                padding: 0.85rem 0.75rem 7.8rem 0.75rem !important;
            }

            .st-key-control_panel,
            .stContainer {
                padding: 16px !important;
                border-radius: 10px !important;
            }

            .nav-text br {
                display: none !important;
            }

            .header-actions {
                align-items: flex-start !important;
            }

            .date-label {
                width: 100% !important;
                text-align: left !important;
                font-size: 0.9rem !important;
            }

            .metric-card,
            .dataset-summary-card {
                min-height: 128px !important;
                padding: 14px 16px !important;
            }

            .metric-value,
            .dataset-summary-value {
                font-size: clamp(1.25rem, 8vw, 1.8rem) !important;
            }

            .metric-tooltip {
                position: static !important;
                opacity: 1 !important;
                transform: none !important;
                pointer-events: auto !important;
                margin-top: 0.75rem !important;
                font-size: 0.8rem !important;
            }

            .metric-tooltip::after {
                display: none !important;
            }

            .ai-insight-card,
            .st-key-ai_insight_card,
            .st-key-backtest_ai_insight_card,
            .variable-info-card,
            .method-card {
                padding: 16px !important;
            }

            .ai-insight-scrollbox,
            .backtest-ai-scrollbox {
                height: 340px !important;
                max-height: 340px !important;
                padding-right: 8px !important;
            }

            .footer-left,
            .footer-right {
                width: 100% !important;
                justify-content: center !important;
                text-align: center !important;
                gap: 0.35rem !important;
            }

            .app-footer {
                font-size: 0.76rem !important;
                gap: 0.45rem !important;
            }

            .live-market-bottom-space {
                height: 0.9rem;
            }
        }

        @media (max-width: 430px) {
            .brand-logo {
                grid-template-columns: 38px minmax(0, 1fr) !important;
            }

            .brand-mark {
                width: 36px !important;
                height: 36px !important;
            }

            .brand-name {
                font-size: 0.9rem !important;
            }

            .nav-link {
                min-height: 44px !important;
                padding: 0.5rem 0.58rem !important;
            }

            .nav-icon {
                width: 1.2rem !important;
                font-size: 0.95rem !important;
            }

            .footer-separator {
                display: none !important;
            }
        }

        .mobile-nav-toggle,
        .mobile-nav-button,
        .mobile-sidebar-backdrop {
            display: none;
        }

        @media (max-width: 1366px) {
            .mobile-nav-toggle {
                position: fixed;
                width: 1px;
                height: 1px;
                opacity: 0;
                pointer-events: none;
            }

            .mobile-nav-button {
                position: fixed;
                top: 16px;
                left: 16px;
                z-index: 2005;
                width: 46px;
                height: 46px;
                display: inline-flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 5px;
                border: 1px solid #4E637D;
                border-radius: 12px;
                background: rgba(38, 53, 69, 0.96);
                box-shadow: 0 10px 24px rgba(5, 10, 18, 0.28);
                cursor: pointer;
            }

            .mobile-nav-button span {
                width: 20px;
                height: 2px;
                border-radius: 999px;
                background: #F8FAFC;
                transition: transform 0.2s ease, opacity 0.2s ease, background-color 0.2s ease;
            }

            .mobile-nav-button:hover span {
                background: #38D9A9;
            }

            body:has(#mobile-nav-toggle:checked) .mobile-nav-button span:nth-child(1) {
                transform: translateY(7px) rotate(45deg);
            }

            body:has(#mobile-nav-toggle:checked) .mobile-nav-button span:nth-child(2) {
                opacity: 0;
            }

            body:has(#mobile-nav-toggle:checked) .mobile-nav-button span:nth-child(3) {
                transform: translateY(-7px) rotate(-45deg);
            }

            .mobile-sidebar-backdrop {
                position: fixed;
                inset: 0;
                z-index: 1998;
                display: block;
                background: rgba(5, 10, 18, 0.55);
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.22s ease;
            }

            body:has(#mobile-nav-toggle:checked) .mobile-sidebar-backdrop {
                opacity: 1;
                pointer-events: auto;
            }

            section[data-testid="stSidebar"] {
                position: fixed !important;
                left: 0 !important;
                top: 0 !important;
                bottom: 0 !important;
                z-index: 2000 !important;
                width: min(84vw, 320px) !important;
                min-width: 0 !important;
                max-width: 320px !important;
                height: 100vh !important;
                transform: translateX(-105%) !important;
                transition: transform 0.24s ease !important;
                border-right: 1px solid #314257 !important;
                border-bottom: 0 !important;
                overflow-y: auto !important;
                box-shadow: 18px 0 36px rgba(5, 10, 18, 0.35) !important;
            }

            body:has(#mobile-nav-toggle:checked) section[data-testid="stSidebar"] {
                transform: translateX(0) !important;
                visibility: visible !important;
            }

            [data-testid="stAppViewContainer"] {
                margin-left: 0 !important;
                width: 100% !important;
            }

            .main .block-container {
                padding-top: 5rem !important;
            }

            .st-key-control_panel [data-testid="stHorizontalBlock"],
            .st-key-control_panel [data-testid="stHorizontalBlock"] > div,
            [data-testid="stHorizontalBlock"],
            [data-testid="stHorizontalBlock"] > div {
                display: flex !important;
                flex-direction: column !important;
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            .button-label-spacer {
                display: none !important;
            }

            .control-help {
                margin-top: -0.2rem !important;
                margin-bottom: 0.35rem !important;
                overflow-wrap: anywhere !important;
            }
        }
    </style>
    """
    return f"<style>{base_css}</style>{overrides}"


def current_language() -> str:
    lang = st.query_params.get("lang", st.session_state.get("language", "id"))
    lang = lang if lang in {"id", "en"} else "id"
    st.session_state.language = lang
    return lang


def t(key: str) -> str:
    lang = current_language()
    return TEXT.get(lang, TEXT["id"]).get(key, TEXT["id"].get(key, key))


def page_href(page_key: str, lang: str | None = None) -> str:
    return f"?page={page_key}&lang={lang or current_language()}"


def localized_date() -> str:
    now = datetime.now(TIMEZONE)
    if current_language() == "en":
        return now.strftime("%A, %d %B %Y")
    try:
        locale.setlocale(locale.LC_TIME, "id_ID.UTF-8")
        return now.strftime("%A, %d %B %Y")
    except locale.Error:
        days = {
            "Monday": "Senin",
            "Tuesday": "Selasa",
            "Wednesday": "Rabu",
            "Thursday": "Kamis",
            "Friday": "Jumat",
            "Saturday": "Sabtu",
            "Sunday": "Minggu",
        }
        months = {
            "January": "Januari",
            "February": "Februari",
            "March": "Maret",
            "April": "April",
            "May": "Mei",
            "June": "Juni",
            "July": "Juli",
            "August": "Agustus",
            "September": "September",
            "October": "Oktober",
            "November": "November",
            "December": "Desember",
        }
        return f"{days[now.strftime('%A')]}, {now.day:02d} {months[now.strftime('%B')]} {now.year}"


def coin_display_options() -> list[str]:
    return [f"{name} ({config['label']})" for name, config in COIN_OPTIONS.items()]


def coin_name_from_display(display_value: str) -> str:
    return display_value.split(" (", 1)[0]


def horizon_options() -> list[str]:
    return ["1D", "3D", "7D"]


def horizon_days(horizon: str | None) -> int:
    if not horizon:
        return 0
    return int(horizon.rstrip("D"))


def forecast_title(days: int) -> str:
    return t("seven_day_title").format(days=days)


def current_page() -> str:
    page = st.query_params.get("page", "prediksi")
    valid_pages = {"prediksi", "komparasi", "variabel", "dataset", "backtesting", "metode"}
    return page if page in valid_pages else "prediksi"


def build_sidebar() -> str:
    selected_page = current_page()
    pages = [
        ("prediksi", "🏠", t("nav_prediction")),
        ("komparasi", "📈", t("nav_comparison")),
        ("backtesting", "↔️", t("nav_backtesting")),
        ("variabel", "❗", t("nav_variable")),
        ("dataset", "🗄️", t("nav_dataset")),
        ("metode", "📚", t("nav_method")),
    ]

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-logo" aria-label="3-Coin Predictor">
                <div class="brand-mark">
                    <span class="brand-orbit orbit-one"></span>
                    <span class="brand-orbit orbit-two"></span>
                    <span class="brand-core">3</span>
                </div>
                <div class="brand-copy">
                    <div class="brand-name"><span>Coin</span><strong>Predictor</strong></div>
                </div>
            </div>
            <div class="sidebar-divider"></div>
            """,
            unsafe_allow_html=True,
        )
        for key, icon, label in pages:
            active = "active" if key == selected_page else ""
            clean_label = label.replace("<br>", " ")
            if active:
                st.markdown(
                    f"""
                    <div class="nav-link active">
                        <span class="nav-icon">{icon}</span>
                        <span class="nav-text">{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif st.button(f"{icon} {clean_label}", key=f"nav_{key}", use_container_width=True):
                st.query_params["page"] = key
                st.query_params["lang"] = current_language()
                st.rerun()

    return selected_page


def render_mobile_nav_toggle() -> None:
    st.markdown(
        """
        <input type="checkbox" id="mobile-nav-toggle" class="mobile-nav-toggle" aria-label="Buka menu navigasi">
        <label for="mobile-nav-toggle" class="mobile-nav-button" aria-label="Buka atau tutup menu navigasi">
            <span></span>
            <span></span>
            <span></span>
        </label>
        <label for="mobile-nav-toggle" class="mobile-sidebar-backdrop" aria-label="Tutup menu navigasi"></label>
        """,
        unsafe_allow_html=True,
    )


def render_page_loader() -> None:
    st.markdown(
        f"""
        <div class="page-loader" aria-hidden="true">
            <div class="page-loader-card">
                <div class="page-loader-mark">
                    <span class="page-loader-core">3</span>
                    <span class="page-loader-orbit orbit-a"></span>
                    <span class="page-loader-orbit orbit-b"></span>
                </div>
                <div class="page-loader-copy">
                    <strong>3-Coin Predictor</strong>
                    <span>{t("page_loading")}</span>
                    <div class="page-loader-bar"><i></i></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_language_switch() -> str:
    page = current_page()
    lang = current_language()
    id_active = "active" if lang == "id" else ""
    en_active = "active" if lang == "en" else ""
    return (
        '<div class="language-switch">'
        f'<a class="{id_active}" href="{page_href(page, "id")}" target="_self">IND</a>'
        f'<a class="{en_active}" href="{page_href(page, "en")}" target="_self">ENG</a>'
        "</div>"
    )


def render_header(title: str | None = None, subtitle: str | None = None) -> None:
    title = title or t("prediction_title")
    subtitle = subtitle or t("prediction_subtitle")
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
            <div class="header-actions">
                {render_language_switch()}
                <div class="date-label">{localized_date()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_metric(name: str, value: float) -> str:
    if pd.isna(value):
        return "-"
    if name in {"MAPE", "Directional Accuracy", "Cumulative Return"}:
        return f"{value:,.2f}%"
    if name == "R2":
        return f"{value:,.3f}"
    return f"{value:,.2f}"


def format_period_date(value: pd.Timestamp) -> str:
    value = pd.to_datetime(value)
    if current_language() == "en":
        return value.strftime("%d %B %Y")
    months = {
        1: "Januari",
        2: "Februari",
        3: "Maret",
        4: "April",
        5: "Mei",
        6: "Juni",
        7: "Juli",
        8: "Agustus",
        9: "September",
        10: "Oktober",
        11: "November",
        12: "Desember",
    }
    return f"{value.day:02d} {months[value.month]} {value.year}"


def render_card_grid(cards: list[str], class_name: str = "") -> None:
    content = "\n".join(cards)
    st.html(f'<div class="responsive-card-grid {class_name}">{content}</div>')


def metric_tooltip(metric_name: str) -> str:
    explanations = {
        "id": {
            "RMSE": "Mengukur rata-rata besar kesalahan prediksi dengan memberi penalti lebih besar pada kesalahan yang jauh. Semakin kecil nilainya, semakin baik.",
            "MAE": "Mengukur rata-rata selisih absolut antara harga aktual dan harga prediksi. Semakin kecil nilainya, semakin dekat prediksi dengan harga aktual.",
            "MAPE": "Mengukur rata-rata persentase kesalahan prediksi terhadap harga aktual. Semakin kecil persentasenya, semakin akurat prediksi.",
            "R2": "Menunjukkan seberapa baik model menjelaskan variasi harga aktual. Nilai yang lebih mendekati 1 berarti model lebih baik.",
            "Directional Accuracy": "Mengukur seberapa sering model berhasil memprediksi arah pergerakan harga, apakah naik atau turun.",
            "Cumulative Return": "Menggambarkan akumulasi imbal hasil dari sinyal prediksi selama periode pengujian.",
        },
        "en": {
            "RMSE": "Measures average prediction error while giving larger penalties to bigger misses. Lower is better.",
            "MAE": "Measures the average absolute difference between actual and predicted prices. Lower means the prediction is closer to the actual price.",
            "MAPE": "Measures average prediction error as a percentage of the actual price. Lower percentage means better accuracy.",
            "R2": "Shows how well the model explains actual price movement. A value closer to 1 means better fit.",
            "Directional Accuracy": "Measures how often the model correctly predicts the price direction, whether up or down.",
            "Cumulative Return": "Shows the accumulated return from prediction-based signals during the testing period.",
        },
    }
    return explanations.get(current_language(), explanations["id"]).get(metric_name, "")


def card_tooltip(card_name: str) -> str:
    explanations = {
        "id": {
            "prediction_reliability": "Menunjukkan kategori ringkas tingkat keandalan prediksi berdasarkan gabungan beberapa metrik evaluasi. Semakin tinggi kategorinya, semakin kuat performa model pada periode tersebut.",
            "confidence_level": "Skor kepercayaan yang dihitung dari Directional Accuracy, R2, dan MAPE. Nilai lebih tinggi berarti model lebih meyakinkan, tetapi tetap bukan jaminan karena pasar kripto sangat volatil.",
            "backtest_period": "Rentang tanggal historis yang digunakan untuk simulasi backtesting, yaitu periode saat harga aktual dibandingkan dengan hasil prediksi model.",
        },
        "en": {
            "prediction_reliability": "A simple reliability category based on combined evaluation metrics. A higher category means the model performed more strongly during that period.",
            "confidence_level": "A confidence score calculated from Directional Accuracy, R2, and MAPE. Higher values mean the model is more convincing, but crypto volatility still makes results uncertain.",
            "backtest_period": "The historical date range used for backtesting, where actual prices are compared against the model prediction.",
        },
    }
    return explanations.get(current_language(), explanations["id"]).get(card_name, "")


def render_metric_grid(metrics: dict[str, float], actual_mean: float | None = None) -> None:
    metric_cards = [
        ("RMSE", "Root Mean Squared Error"),
        ("MAE", "Mean Absolute Error"),
        ("MAPE", "Mean Absolute Percentage Error"),
        ("R2", "Coefficient of Determination"),
        ("Directional Accuracy", t("metric_directional")),
        ("Cumulative Return", t("metric_return")),
    ]

    cards = [
        f"""
        <div class="metric-card {metric_quality_class(key, metrics[key], actual_mean or 0)}">
            <div class="metric-label">{key}</div>
            <div class="metric-value">{format_metric(key, metrics[key])}</div>
            <div class="metric-help">{helper}</div>
            <div class="metric-tooltip">{html.escape(metric_tooltip(key))}</div>
        </div>
        """
        for key, helper in metric_cards
    ]
    render_card_grid(cards, "metric-card-grid")


def metric_quality_class(metric_name: str, value: float, actual_mean: float) -> str:
    if pd.isna(value):
        return "metric-neutral"
    if metric_name in {"RMSE", "MAE"}:
        ratio = abs(value) / actual_mean if actual_mean else 1
        if ratio <= 0.03:
            return "metric-good"
        if ratio <= 0.07:
            return "metric-medium"
        return "metric-bad"
    if metric_name == "MAPE":
        if value <= 3:
            return "metric-good"
        if value <= 7:
            return "metric-medium"
        return "metric-bad"
    if metric_name == "R2":
        if value >= 0.8:
            return "metric-good"
        if value >= 0.5:
            return "metric-medium"
        return "metric-bad"
    if metric_name == "Directional Accuracy":
        if value >= 70:
            return "metric-good"
        if value >= 55:
            return "metric-medium"
        return "metric-bad"
    if metric_name == "Cumulative Return":
        if value > 1:
            return "metric-good"
        if value >= -1:
            return "metric-medium"
        return "metric-bad"
    return "metric-neutral"


def calculate_prediction_confidence(metrics: dict[str, float]) -> tuple[float, str]:
    directional = float(metrics.get("Directional Accuracy", 0) or 0)
    r2_value = float(metrics.get("R2", 0) or 0)
    mape = float(metrics.get("MAPE", 100) or 100)

    if pd.isna(directional):
        directional = 0
    if pd.isna(r2_value):
        r2_value = 0
    if pd.isna(mape):
        mape = 100

    directional = max(0, min(100, directional))
    normalized_r2 = max(0, min(100, r2_value * 100))
    inverse_mape = max(0, min(100, 100 - max(0, mape)))
    confidence = (directional * 0.4) + (normalized_r2 * 0.3) + (inverse_mape * 0.3)

    if confidence >= 80:
        level = "High"
    elif confidence >= 60:
        level = "Medium"
    else:
        level = "Low"
    return float(confidence), level


def confidence_level_label(level: str) -> str:
    if current_language() == "en":
        return level
    return {"High": "Tinggi", "Medium": "Sedang", "Low": "Rendah"}.get(level, level)


def confidence_helper_text() -> str:
    if current_language() == "en":
        return "Based on Directional Accuracy, R2, and MAPE"
    return "Berdasarkan Directional Accuracy, R2, dan MAPE"


def render_prediction_confidence_cards(result: dict[str, object]) -> None:
    confidence_score, confidence_level = calculate_prediction_confidence(result["metrics"])
    level_label = confidence_level_label(confidence_level)
    cards = [
        f"""
        <div class="metric-card {confidence_card_class(confidence_level)}">
            <div class="metric-label">{t("prediction_reliability")}</div>
            <div class="metric-value">{level_label}</div>
            <div class="metric-help">{confidence_helper_text()}</div>
            <div class="metric-tooltip">{html.escape(card_tooltip("prediction_reliability"))}</div>
        </div>
        """,
        f"""
        <div class="metric-card {confidence_card_class(confidence_level)}">
            <div class="metric-label">{t("confidence_level")}</div>
            <div class="metric-value">{confidence_score:.2f}%</div>
            <div class="metric-help">High 80-100 | Medium 60-79 | Low &lt;60</div>
            <div class="metric-tooltip">{html.escape(card_tooltip("confidence_level"))}</div>
        </div>
        """,
    ]
    render_card_grid(cards, "forecast-confidence-grid")


def render_backtest_metric_grid(metrics: dict[str, float], actual_mean: float) -> None:
    metric_cards = [
        ("RMSE", "Root Mean Squared Error"),
        ("MAE", "Mean Absolute Error"),
        ("MAPE", "Mean Absolute Percentage Error"),
        ("R2", "Coefficient of Determination"),
        ("Directional Accuracy", t("metric_directional")),
        ("Cumulative Return", t("metric_return")),
    ]
    cards = [
        f"""
        <div class="metric-card {metric_quality_class(key, metrics[key], actual_mean)}">
            <div class="metric-label">{key}</div>
            <div class="metric-value">{format_metric(key, metrics[key])}</div>
            <div class="metric-help">{helper}</div>
        </div>
        """
        for key, helper in metric_cards
    ]
    render_card_grid(cards, "metric-card-grid")


def prediction_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    price_hover = (
        "<b>%{fullData.name}</b><br>"
        "📅 %{x|%d %b %Y}<br>"
        "💰 $%{y:,.2f}"
        "<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Actual Price"],
            mode="lines",
            name=t("actual_price"),
            line=dict(color="#4DB8FF", width=2.6),
            hovertemplate=price_hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Predicted Price"],
            mode="lines",
            name=t("predicted_price"),
            line=dict(color="#38D9A9", width=2.6),
            hovertemplate=price_hover,
        )
    )
    return style_chart(fig, title)


def forecast_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    price_hover = (
        "<b>%{fullData.name}</b><br>"
        "📅 %{x|%d %b %Y}<br>"
        "💰 $%{y:,.2f}"
        "<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Historical Price"],
            mode="lines",
            name=t("historical_price"),
            line=dict(color="#4DB8FF", width=2.4),
            connectgaps=False,
            hovertemplate=price_hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Predicted Price"],
            mode="lines+markers",
            name=t("seven_day_prediction"),
            line=dict(color="#38D9A9", width=2.8),
            marker=dict(size=7),
            connectgaps=False,
            hovertemplate=price_hover,
        )
    )
    return style_chart(fig, title)


def backtest_chart(frame: pd.DataFrame, title: str, cutoff_date: pd.Timestamp) -> go.Figure:
    price_hover = (
        "<b>%{fullData.name}</b><br>"
        "📅 %{x|%d %b %Y}<br>"
        "💰 $%{y:,.2f}"
        "<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Historical Price"],
            mode="lines",
            name=t("historical_price"),
            line=dict(color="#4DB8FF", width=2.4),
            connectgaps=False,
            hovertemplate=price_hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Actual Future Price"],
            mode="lines+markers",
            name=t("actual_price"),
            line=dict(color="#F8FAFC", width=2.6),
            marker=dict(size=7),
            connectgaps=False,
            hovertemplate=price_hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Predicted Price"],
            mode="lines+markers",
            name=t("predicted_price"),
            line=dict(color="#38D9A9", width=2.8, dash="dash"),
            marker=dict(size=7),
            connectgaps=False,
            hovertemplate=price_hover,
        )
    )
    fig.add_vline(x=pd.to_datetime(cutoff_date), line_width=1, line_dash="dot", line_color="#C0CAD8")
    return style_chart(fig, title)


def get_gemini_api_key() -> str:
    try:
        secret_value = st.secrets.get("GEMINI_API_KEY", "")
        if secret_value:
            return str(secret_value)
        gemini_section = st.secrets.get("gemini", {})
        if isinstance(gemini_section, dict) and gemini_section.get("api_key"):
            return str(gemini_section["api_key"])
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.getenv("GEMINI_API_KEY", "")


def get_gemini_model_name() -> str:
    try:
        secret_value = st.secrets.get("GEMINI_MODEL", "")
        if secret_value:
            return str(secret_value)
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def get_ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


@st.cache_data(ttl=5, show_spinner=False)
def fetch_live_market_overview() -> list[dict[str, object]]:
    rows = []
    symbol_map = {
        "Bitcoin": ("BTCUSDT", "₿", "BTC"),
        "Ethereum": ("ETHUSDT", "Ξ", "ETH"),
        "Solana": ("SOLUSDT", "◎", "SOL"),
    }
    for coin_name, (symbol, icon, ticker) in symbol_map.items():
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 3-Coin-Predictor/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=6, context=get_ssl_context()) as response:
                data = json.loads(response.read().decode("utf-8"))
            change = float(data.get("priceChangePercent", 0))
            price = float(data.get("lastPrice", 0))
            if change > 1:
                signal = "Bullish"
                signal_class = "signal-bullish"
            elif change < -1:
                signal = "Bearish"
                signal_class = "signal-bearish"
            else:
                signal = "Neutral"
                signal_class = "signal-neutral"
            rows.append(
                {
                    "coin": coin_name,
                    "ticker": ticker,
                    "icon": icon,
                    "price": price,
                    "change": change,
                    "signal": signal,
                    "signal_class": signal_class,
                }
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            rows.append(
                {
                    "coin": coin_name,
                    "ticker": ticker,
                    "icon": icon,
                    "price": None,
                    "change": None,
                    "signal": "-",
                    "signal_class": "signal-neutral",
                }
            )
    return rows


@st.cache_data(ttl=5, show_spinner=False)
def fetch_live_fear_greed() -> dict[str, object] | None:
    url = "https://api.alternative.me/fng/?limit=1"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 3-Coin-Predictor/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=6, context=get_ssl_context()) as response:
            payload = json.loads(response.read().decode("utf-8"))
        item = payload["data"][0]
        value = int(item["value"])
        classification = str(item.get("value_classification", ""))
        if value <= 24:
            band = "fear-extreme"
        elif value <= 44:
            band = "fear"
        elif value <= 55:
            band = "neutral"
        elif value <= 74:
            band = "greed"
        else:
            band = "greed-extreme"
        return {"value": value, "classification": classification, "band": band}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_crypto_news_context(coin_name: str) -> list[dict[str, str]]:
    query = urllib.parse.quote(f"{coin_name} crypto market sentiment global economy when:10d")
    url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 3-Coin-Predictor/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8, context=get_ssl_context()) as response:
            xml_text = response.read()
        root = ET.fromstring(xml_text)
        headlines = []
        now = pd.Timestamp.now(tz=TIMEZONE)
        cutoff = now - pd.Timedelta(days=10)
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            source = item.findtext("source", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            published_at = pd.to_datetime(pub_date, utc=True, errors="coerce")
            if pd.isna(published_at):
                continue
            published_at = published_at.tz_convert(TIMEZONE)
            if published_at < cutoff or published_at > now + pd.Timedelta(days=1):
                continue
            if title and link:
                headlines.append(
                    {
                        "title": title,
                        "source": source,
                        "url": link,
                        "published_at": published_at.isoformat(),
                    }
                )
            if len(headlines) >= 6:
                break
        return headlines
    except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError):
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_crypto_news_context_for_period(
    coin_name: str,
    period_start: object,
    period_end: object,
    tolerance_days: int = 10,
) -> list[dict[str, str]]:
    start_ts = pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(TIMEZONE)
    else:
        start_ts = start_ts.tz_convert(TIMEZONE)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(TIMEZONE)
    else:
        end_ts = end_ts.tz_convert(TIMEZONE)

    window_start = start_ts.normalize() - pd.Timedelta(days=tolerance_days)
    window_end = end_ts.normalize() + pd.Timedelta(days=tolerance_days + 1)
    query_start = window_start.strftime("%Y-%m-%d")
    query_end = window_end.strftime("%Y-%m-%d")
    query = urllib.parse.quote(
        f"{coin_name} crypto market sentiment global economy after:{query_start} before:{query_end}"
    )
    url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 3-Coin-Predictor/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8, context=get_ssl_context()) as response:
            xml_text = response.read()
        root = ET.fromstring(xml_text)
        headlines = []
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            source = item.findtext("source", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            published_at = pd.to_datetime(pub_date, utc=True, errors="coerce")
            if pd.isna(published_at):
                continue
            published_at = published_at.tz_convert(TIMEZONE)
            if published_at < window_start or published_at >= window_end:
                continue
            if title and link:
                headlines.append(
                    {
                        "title": title,
                        "source": source,
                        "url": link,
                        "published_at": published_at.isoformat(),
                    }
                )
            if len(headlines) >= 6:
                break
        return headlines
    except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError):
        return []


def build_ai_prompt(result: dict, news_context: list[dict[str, str]]) -> str:
    forecast_rows = result["forecast_table"].copy()
    forecast_rows["Date"] = pd.to_datetime(forecast_rows["Date"]).dt.strftime("%Y-%m-%d")
    forecast_payload = forecast_rows.to_dict(orient="records")
    forecast_days_value = int(result.get("forecast_days", len(forecast_rows)))
    news_payload = [
        {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "tanggal": item.get("published_at", "")[:10],
        }
        for item in news_context
    ] or [{"title": "Tidak ada referensi berita valid dalam 10 hari terakhir.", "source": "", "tanggal": ""}]

    if current_language() == "en":
        return f"""
You are a crypto market analyst for general users of the 3-Coin Predictor dashboard.
Write in clear, public-friendly English.
Main focus: explain the {forecast_days_value}-day forecast result and why the price may move that way.
Avoid technical statistics/CS terms such as RMSE, MAE, R2, MAPE, feature, testing, and directional accuracy.
If model performance must be mentioned, simply say "the model looks fairly stable/less stable" without technical numbers.
Do not give buy/sell instructions. Length: 430-620 words.
Start directly with the title "## 1. Forecast Result Analysis".
Do not write an opening greeting, filler, or conversational phrases such as "Sure", "Let's", "Here is", "I will", or "for you".

Coin: {result["coin_name"]}
Prediction model: {result["model_name"]}
{forecast_days_value}-day forecast: {json.dumps(forecast_payload, ensure_ascii=False)}
Latest news/sentiment context from the last 10 days: {json.dumps(news_payload, ensure_ascii=False)}

Use this exact format:
## 1. Forecast Result Analysis
Write 1-2 paragraphs. Explain whether the next {forecast_days_value} day(s) look upward, downward, or range-bound. Mention the start, peak/low, and final forecast if visible from the data.

## 2. Why Might the Forecast Look This Way?
Write 1 paragraph. Explain the possible reason in simple language, such as momentum, correction after a rise, cautious movement, or consolidation.

## 3. Market Conditions and Global Sentiment
Write 1-2 paragraphs. Connect the forecast with available news/global sentiment, economic conditions, investor interest in crypto, interest-rate/inflation/US dollar/regulation/geopolitical risks if relevant. Explain the potential impact on the selected coin.

## 4. Key Things to Watch
Write 1 paragraph with simple risk notes. Clearly explain that crypto prices are very volatile, so the forecast may differ from actual future prices. Mention that the prediction model is limited to historical data, statistical patterns, machine learning, and available external variables such as volume, volatility, and Fear & Greed. Explain that the model itself cannot truly understand breaking news, sudden market sentiment changes, regulatory shocks, geopolitical events, or unexpected volatility. End with: "Not financial advice."
"""

    return f"""
Anda adalah analis pasar kripto untuk pengguna umum dashboard 3-Coin Predictor.
Tulis dalam Bahasa Indonesia yang mudah dipahami publik umum.
Fokus utama: jelaskan hasil forecast {forecast_days_value} hari ke depan dan alasan kemungkinan pergerakannya.
Hindari istilah teknis statistik/CS seperti RMSE, MAE, R2, MAPE, feature, testing, dan directional accuracy.
Jika perlu menyebut performa model, cukup sebut "model terlihat cukup stabil/kurang stabil" tanpa angka teknis.
Jangan memberikan instruksi beli/jual. Panjang jawaban 430-620 kata.
Langsung mulai dari judul "## 1. Analisis Hasil Forecast".
Jangan menulis kalimat pembuka, sapaan, filler, atau frasa percakapan seperti "Tentu", "Mari kita", "Berikut adalah", "Saya akan", atau "Untuk Anda".

Koin: {result["coin_name"]}
Model prediksi: {result["model_name"]}
Forecast {forecast_days_value} hari: {json.dumps(forecast_payload, ensure_ascii=False)}
Konteks berita/sentimen terbaru maksimal 10 hari terakhir: {json.dumps(news_payload, ensure_ascii=False)}

WAJIB gunakan format berikut persis:
## 1. Analisis Hasil Forecast
Tulis 1-2 paragraf. Jelaskan arah harga {forecast_days_value} hari ke depan, apakah cenderung naik, turun, atau bergerak terbatas. Sebutkan awal, puncak/lembah, dan akhir forecast jika terlihat dari data.

## 2. Mengapa Prediksinya Seperti Itu?
Tulis 1 paragraf. Jelaskan dengan bahasa sederhana kemungkinan alasan pola forecast, misalnya momentum harga, koreksi setelah kenaikan, atau pergerakan yang masih hati-hati.

## 3. Kondisi Market dan Sentimen Global
Tulis 1-2 paragraf. Hubungkan dengan berita/sentimen global yang tersedia, kondisi ekonomi, minat investor pada kripto, risiko suku bunga/inflasi/dolar AS/regulasi/geopolitik jika relevan. Jelaskan dampaknya pada koin terpilih.

## 4. Hal yang Perlu Diperhatikan
Tulis 1 paragraf berisi risiko dengan bahasa sederhana. Jelaskan dengan jelas bahwa harga kripto sangat volatil, sehingga hasil forecast bisa berbeda dari harga aktual di masa depan. Sebutkan bahwa model prediksi terbatas pada data historis, pola statistik, machine learning, serta variabel eksternal yang tersedia seperti volume, volatilitas, dan Fear & Greed. Jelaskan bahwa model tidak benar-benar bisa memahami berita mendadak, perubahan sentimen pasar tiba-tiba, regulasi, kondisi geopolitik, atau lonjakan volatilitas yang tidak terduga. Akhiri dengan kalimat: "Bukan nasihat finansial."
"""


def build_backtest_ai_prompt(result: dict, news_context: list[dict[str, str]]) -> str:
    table = result["comparison_table"].copy()
    table["Date"] = pd.to_datetime(table["Date"]).dt.strftime("%Y-%m-%d")
    table_payload = table[["Date", "Actual Price", "Predicted Price", "Error", "Trend"]].to_dict(orient="records")
    metrics_payload = {key: format_metric(key, value) for key, value in result["metrics"].items()}
    news_payload = [
        {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "date": item.get("published_at", "")[:10],
        }
        for item in news_context
    ] or [{"title": "No valid news references found for the selected period.", "source": "", "date": ""}]

    if current_language() == "en":
        return f"""
Write a public-friendly backtesting insight in English.
Do not use greetings or conversational filler. Do not give buy/sell advice.
Use short bullet points only.

Coin: {result["coin_name"]}
Model: {result["model_name"]}
Backtesting table: {json.dumps(table_payload, ensure_ascii=False)}
Metrics: {json.dumps(metrics_payload, ensure_ascii=False)}
Confidence: {result["confidence_score"]:.2f}% ({result["confidence_level"]})
News context: {json.dumps(news_payload, ensure_ascii=False)}

Use exactly this format:
## Model Analysis
• 2-3 bullets about whether the model captured the price direction and where it lagged or missed.

## Market Sentiment
• 2-3 bullets about Fear & Greed, volume, volatility, and market/news conditions around that period if available.

## Practical Meaning
• 2-3 bullets explaining what a public user should understand from this backtest.
• Include one bullet explaining that crypto is highly volatile, so backtesting differences can happen because the model is limited to historical data, statistical/machine-learning patterns, and available external variables such as volume, volatility, and Fear & Greed. Explain that the model itself cannot truly understand breaking news, sudden sentiment shifts, regulatory shocks, geopolitical events, or unexpected volatility.
"""

    return f"""
Tulis insight backtesting dalam Bahasa Indonesia yang mudah dipahami publik.
Jangan gunakan sapaan atau filler percakapan. Jangan memberi saran beli/jual.
Gunakan bullet pendek saja.

Koin: {result["coin_name"]}
Model: {result["model_name"]}
Tabel backtesting: {json.dumps(table_payload, ensure_ascii=False)}
Metrik: {json.dumps(metrics_payload, ensure_ascii=False)}
Confidence: {result["confidence_score"]:.2f}% ({result["confidence_level"]})
Konteks berita: {json.dumps(news_payload, ensure_ascii=False)}

Gunakan format persis ini:
## Analisis Model
• 2-3 bullet tentang apakah model menangkap arah harga dan bagian mana yang terlambat atau meleset.

## Sentimen Market
• 2-3 bullet tentang Fear & Greed, volume, volatilitas, serta kondisi market/berita pada periode itu jika tersedia.

## Makna Praktis
• 2-3 bullet tentang apa yang perlu dipahami pengguna umum dari hasil backtest ini.
• Wajib sertakan satu bullet yang menjelaskan bahwa kripto sangat volatil, sehingga selisih backtesting bisa terjadi karena model terbatas pada data historis, pola statistik/machine learning, dan variabel eksternal yang tersedia seperti volume, volatilitas, dan Fear & Greed. Jelaskan bahwa model tidak benar-benar bisa memahami berita mendadak, perubahan sentimen pasar tiba-tiba, regulasi, geopolitik, atau volatilitas yang tidak terduga.
"""


@st.cache_data(ttl=1800, show_spinner=False)
def request_gemini_insight(prompt: str, api_key: str, model_name: str, cache_version: str) -> str:
    candidate_models = []
    for candidate in [model_name, "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        if candidate and candidate not in candidate_models:
            candidate_models.append(candidate)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.35,
            "topP": 0.9,
            "maxOutputTokens": 1536,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    errors = []
    for candidate_model in candidate_models:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(candidate_model)}:generateContent?key={urllib.parse.quote(api_key)}"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=25, context=get_ssl_context()) as response:
                data = json.loads(response.read().decode("utf-8"))
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            text_parts = [part.get("text", "") for part in parts if part.get("text")]
            insight_text = "\n".join(text_parts).strip()
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if insight_text:
                if finish_reason not in {"STOP", "UNKNOWN"}:
                    insight_text += f"\n\nCatatan teknis: respons Gemini berhenti dengan status {finish_reason}."
                return insight_text
            errors.append(f"{candidate_model}: Gemini mengembalikan respons kosong.")
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
            errors.append(f"{candidate_model}: {exc}")

    return "Insight AI belum dapat dimuat. Detail teknis: " + " | ".join(errors)


def insight_text_to_html(text: str, require_numbered_start: bool = True) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    while require_numbered_start and lines and not re.match(r"^(#{1,6}\s*)?1\.\s+", lines[0]):
        lines.pop(0)
    if not lines:
        return "<p>Insight AI belum tersedia.</p>"
    html_lines = []
    for line in lines:
        safe_line = html.escape(line)
        safe_line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", safe_line)
        had_heading_marker = bool(re.match(r"^#{1,6}\s*", safe_line))
        safe_line = re.sub(r"^#{1,6}\s*", "", safe_line)
        heading_match = re.match(r"^(\d+)\.\s+(.+)$", safe_line)
        if heading_match or had_heading_marker:
            heading_text = f"{heading_match.group(1)}. {heading_match.group(2)}" if heading_match else safe_line
            html_lines.append(f'<p class="ai-section-heading">{heading_text}</p>')
            continue
        line_class = "ai-bullet" if safe_line.startswith(("-", "*")) else ""
        if line_class:
            html_lines.append(f"<p class='{line_class}'>{safe_line}</p>")
        else:
            html_lines.append(f"<p>{safe_line}</p>")
    return "\n".join(html_lines)


def news_references_to_html(news_context: list[dict[str, str]], title: str | None = None) -> str:
    valid_items = [item for item in news_context if item.get("title") and item.get("url")]
    if not valid_items:
        return ""

    links = []
    for index, item in enumerate(valid_items[:5], start=1):
        title = html.escape(item["title"])
        source = html.escape(item.get("source", "Sumber berita"))
        published_at = item.get("published_at", "")
        date_label = ""
        if published_at:
            date_label = f" • {html.escape(format_period_date(pd.Timestamp(published_at)))}"
        url = html.escape(item["url"], quote=True)
        links.append(
            (
                f'<a class="ai-news-reference" href="{url}" target="_blank" rel="noopener noreferrer">'
                f"<span>{index}. {title}</span>"
                f"<small>{source}{date_label}</small>"
                "</a>"
            )
        )
    return (
        '<div class="ai-reference-block">'
        f'<p class="ai-reference-title">{t("ai_refs")}</p>'
        f"{''.join(links)}"
        "</div>"
    )


def render_ai_insight(result: dict) -> None:
    api_key = get_gemini_api_key()
    model_name = get_gemini_model_name()

    if not api_key:
        with st.container(key="ai_insight_card"):
            st.markdown(
                (
                    '<div class="ai-insight-title">'
                    '<span class="ai-insight-icon">💡</span>'
                    "<span>AI Insight</span>"
                    "</div>"
                    '<div class="ai-insight-divider"></div>'
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                (
                    f'<p>{t("ai_missing")}</p>'
                ),
                unsafe_allow_html=True,
            )
        return

    with st.spinner(t("ai_loading")):
        news_context = fetch_crypto_news_context(result["coin_name"])
        prompt = build_ai_prompt(result, news_context)
        insight = request_gemini_insight(prompt, api_key, model_name, AI_INSIGHT_CACHE_VERSION)

    with st.container(key="ai_insight_card"):
        st.markdown(
            (
                '<div class="ai-insight-title">'
                '<span class="ai-insight-icon">💡</span>'
                "<span>AI Insight</span>"
                "</div>"
                '<div class="ai-insight-divider"></div>'
            ),
            unsafe_allow_html=True,
        )
        insight_content = insight_text_to_html(insight)
        reference_content = news_references_to_html(news_context)
        st.markdown(
            f'<div class="ai-insight-scrollbox">{insight_content}{reference_content}</div>',
            unsafe_allow_html=True,
        )


def render_backtest_ai_insight(result: dict) -> None:
    api_key = get_gemini_api_key()
    model_name = get_gemini_model_name()
    if not api_key:
        st.markdown(f'<div class="placeholder">{t("ai_missing")}</div>', unsafe_allow_html=True)
        return

    with st.spinner(t("backtest_ai_loading")):
        period_start, period_end = result["backtest_period"]
        context_start = result.get("cutoff_date", period_start)
        news_context = fetch_crypto_news_context_for_period(result["coin_name"], context_start, period_end)
        prompt = build_backtest_ai_prompt(result, news_context)
        insight = request_gemini_insight(prompt, api_key, model_name, f"{AI_INSIGHT_CACHE_VERSION}-backtest")

    with st.container(key="backtest_ai_insight_card"):
        st.markdown(
            (
                '<div class="ai-insight-title">'
                '<span class="ai-insight-icon">💡</span>'
                "<span>AI Insight</span>"
                "</div>"
                '<div class="ai-insight-divider"></div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="ai-insight-scrollbox backtest-ai-scrollbox">{insight_text_to_html(insight, require_numbered_start=False)}{news_references_to_html(news_context, t("backtest_ai_refs"))}</div>',
            unsafe_allow_html=True,
        )


def feature_importance_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    ordered = frame.sort_values("Importance", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ordered["Importance"],
            y=ordered["Feature"],
            orientation="h",
            marker=dict(
                color=ordered["Importance"],
                colorscale=[[0, "#4DB8FF"], [1, "#38D9A9"]],
                line=dict(color="#38D9A9", width=1),
            ),
            hovertemplate="<b>%{y}</b><br>📊 Importance Score<br>%{x:.4f}<extra></extra>",
            name="Importance",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=430,
        margin=dict(l=22, r=22, t=58, b=22),
        paper_bgcolor="#263545",
        plot_bgcolor="#182231",
        font=dict(color="#F8FAFC", family="Inter, Segoe UI, Arial, sans-serif"),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#263545",
            bordercolor="#38D9A9",
            font=dict(color="#F8FAFC", size=16, family="Inter, Segoe UI, Arial, sans-serif"),
            align="left",
            namelength=-1,
        ),
    )
    fig.update_xaxes(title_text="Importance Score", gridcolor="#425168", zeroline=False)
    fig.update_yaxes(title_text="", gridcolor="#425168", zeroline=False)
    return fig


def residual_distribution_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    residuals = frame["Residual"].dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=32,
            marker=dict(color="#4DB8FF", line=dict(color="#38D9A9", width=1)),
            opacity=0.82,
            name="Residual",
            hovertemplate="Residual: %{x:.4f}<br>Frekuensi: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#C0CAD8")
    styled = style_chart(fig, title)
    styled.update_layout(showlegend=False, height=330)
    styled.update_xaxes(title_text="Residual")
    styled.update_yaxes(title_text="Frequency")
    return styled


def residual_autocorrelation_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    clean_frame = frame.dropna(subset=["Autocorrelation"]).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=clean_frame["Lag"],
            y=clean_frame["Autocorrelation"],
            marker_color=[
                "#FF6B7A" if abs(value) >= 0.2 else "#38D9A9"
                for value in clean_frame["Autocorrelation"]
            ],
            hovertemplate="Lag %{x}<br>Autocorrelation: %{y:.4f}<extra></extra>",
            name="Autocorrelation",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#C0CAD8")
    fig.add_hline(y=0.2, line_width=1, line_dash="dot", line_color="#FFD166")
    fig.add_hline(y=-0.2, line_width=1, line_dash="dot", line_color="#FFD166")
    styled = style_chart(fig, title)
    styled.update_layout(showlegend=False, height=330)
    styled.update_xaxes(title_text="Lag", dtick=1)
    styled.update_yaxes(title_text="Autocorrelation", range=[-1, 1])
    return styled


def render_sarimax_diagnostics(result: dict[str, object]) -> None:
    diagnostics = result.get("residual_diagnostics")
    if not diagnostics:
        return

    ljung_box = diagnostics.get("ljung_box", {})
    p_value = float(ljung_box.get("p_value", float("nan")))
    interpretation_key = "ljung_box_pass" if p_value >= 0.05 else "ljung_box_pattern"
    conclusion_key = "residual_conclusion_good" if p_value >= 0.05 else "residual_conclusion_watch"
    conclusion_helper_key = "residual_conclusion_helper_good" if p_value >= 0.05 else "residual_conclusion_helper_watch"
    quality_class = "metric-good" if p_value >= 0.05 else "metric-medium"
    lag = ljung_box.get("lag", "-")

    st.markdown(
        f'<div class="section-title" style="margin-top: 1.5rem;">{t("sarimax_diagnostics_title")}</div>',
        unsafe_allow_html=True,
    )
    cards = [
        f"""
        <div class="metric-card {quality_class}">
            <div class="metric-label">{t("ljung_box_title")}</div>
            <div class="metric-value">{t(interpretation_key)}</div>
            <div class="metric-help">{t("ljung_box_pvalue")}: {p_value:.4f} | {t("ljung_box_lag")}: {lag}</div>
            <div class="metric-tooltip">{html.escape(t("ljung_box_tooltip"))}</div>
        </div>
        """,
        f"""
        <div class="metric-card {quality_class}">
            <div class="metric-label">{t("residual_conclusion")}</div>
            <div class="metric-value dataset-summary-value">{t(conclusion_key)}</div>
            <div class="metric-help">{t(conclusion_helper_key)}</div>
            <div class="metric-tooltip">{html.escape(t("residual_conclusion_tooltip"))}</div>
        </div>
        """,
    ]
    render_card_grid(cards, "forecast-confidence-grid")

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        st.plotly_chart(
            residual_distribution_chart(
                diagnostics["residuals"],
                f"{result['coin_name']} - {t('residual_distribution_title')} (SARIMAX)",
            ),
            use_container_width=True,
        )
    with chart_cols[1]:
        st.plotly_chart(
            residual_autocorrelation_chart(
                diagnostics["autocorrelation"],
                f"{result['coin_name']} - {t('residual_autocorrelation_title')} (SARIMAX)",
            ),
            use_container_width=True,
        )
    st.markdown(f'<div class="chart-caption">{t("sarimax_diagnostics_caption")}</div>', unsafe_allow_html=True)


def style_chart(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        margin=dict(l=22, r=22, t=58, b=22),
        legend=dict(orientation="h", x=0, y=1.08),
        paper_bgcolor="#263545",
        plot_bgcolor="#182231",
        font=dict(color="#F8FAFC", family="Inter, Segoe UI, Arial, sans-serif"),
        hoverlabel=dict(
            bgcolor="#263545",
            bordercolor="#38D9A9",
            font=dict(color="#F8FAFC", size=16, family="Inter, Segoe UI, Arial, sans-serif"),
            align="left",
            namelength=-1,
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#425168", zeroline=False)
    return fig


def style_comparison_chart(fig: go.Figure) -> go.Figure:
    fig = style_chart(fig, "")
    fig.update_layout(
        height=320,
        margin=dict(l=18, r=18, t=18, b=20),
        legend=dict(orientation="h", x=0, y=1.04),
    )
    return fig


@st.cache_data(ttl=1800, show_spinner=False)
def run_model_comparison(coin_name: str, data_path: str) -> dict[str, object]:
    results = {
        model_name: run_prediction(
            coin_name=coin_name,
            model_name=model_name,
            data_path=data_path,
            days=7,
        )
        for model_name in ["SARIMAX", "XGBoost"]
    }

    rows = []
    for model_name, result in results.items():
        metrics = result["metrics"]
        confidence_score, confidence_level = calculate_prediction_confidence(metrics)
        rows.append(
            {
                "Koin": COIN_OPTIONS[coin_name]["label"],
                "Model": model_name,
                "RMSE": metrics.get("RMSE"),
                "MAE": metrics.get("MAE"),
                "MAPE": metrics.get("MAPE"),
                "R2": metrics.get("R2"),
                "Directional Accuracy": metrics.get("Directional Accuracy"),
                "Cumulative Return": metrics.get("Cumulative Return"),
                "Confidence Score": confidence_score,
                "Reliability": confidence_level,
            }
        )

    comparison_table = pd.DataFrame(rows)
    best_row = (
        comparison_table.sort_values(
            by=["Confidence Score", "MAPE", "RMSE"],
            ascending=[False, True, True],
        )
        .iloc[0]
        .to_dict()
    )

    return {
        "coin_name": coin_name,
        "results": results,
        "comparison_table": comparison_table,
        "best_row": best_row,
        "test_period": results["SARIMAX"].get("test_period"),
    }


def render_comparison_best_cards(result: dict[str, object] | None) -> None:
    if result is None:
        coin_value = "-"
        model_value = "-"
    else:
        best = result["best_row"]
        coin_value = str(best.get("Koin", "-"))
        model_value = str(best.get("Model", "-"))

    st.html(
        f"""
        <section class="comparison-best-card">
            <div class="comparison-best-title">{t("comparison_best_title")}</div>
            <div class="comparison-best-grid">
                <div class="metric-card">
                    <div class="metric-label">{t("comparison_best_coin")}</div>
                    <div class="metric-value">{html.escape(coin_value)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">{t("comparison_best_model")}</div>
                    <div class="metric-value">{html.escape(model_value)}</div>
                </div>
            </div>
        </section>
        """
    )


def format_comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.drop(columns=["Koin"], errors="ignore").copy()
    for column in ["RMSE", "MAE"]:
        display[column] = display[column].map(lambda value: f"{value:,.2f}" if pd.notna(value) else "-")
    for column in ["MAPE", "Directional Accuracy", "Cumulative Return", "Confidence Score"]:
        display[column] = display[column].map(lambda value: f"{value:,.2f}%" if pd.notna(value) else "-")
    display["R2"] = display["R2"].map(lambda value: f"{value:,.3f}" if pd.notna(value) else "-")
    display["Reliability"] = display["Reliability"].map(confidence_level_label)
    return display


def comparison_error_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"SARIMAX": "#4DB8FF", "XGBoost": "#38D9A9"}
    for model_name in frame["Model"]:
        row = frame[frame["Model"] == model_name].iloc[0]
        fig.add_trace(
            go.Bar(
                x=["RMSE", "MAE", "MAPE"],
                y=[row["RMSE"], row["MAE"], row["MAPE"]],
                name=model_name,
                marker_color=colors.get(model_name, "#C0CAD8"),
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(type="log", title_text="Log scale")
    return style_comparison_chart(fig)


def comparison_direction_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["Model"],
            y=frame["Directional Accuracy"],
            marker_color=["#4DB8FF" if model == "SARIMAX" else "#38D9A9" for model in frame["Model"]],
            hovertemplate="<b>%{x}</b><br>Directional Accuracy: %{y:,.2f}%<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return style_comparison_chart(fig)


def comparison_return_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["Cumulative Return"],
            y=frame["Model"],
            orientation="h",
            marker_color=["#FF6B7A" if value < 0 else "#38D9A9" for value in frame["Cumulative Return"]],
            marker_line=dict(color=["rgba(255,107,122,0.65)" if value < 0 else "rgba(56,217,169,0.65)" for value in frame["Cumulative Return"]], width=1),
            hovertemplate="<b>%{y}</b><br>Cumulative Return: %{x:,.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#C0CAD8")
    fig.update_xaxes(ticksuffix="%")
    return style_comparison_chart(fig)


def build_comparison_ai_prompt(result: dict[str, object]) -> str:
    table = result["comparison_table"].copy()
    payload = table.to_dict(orient="records")
    best = result["best_row"]
    if current_language() == "en":
        return f"""
Write a public-friendly model comparison insight in English.
Do not use greetings or conversational filler. Do not give buy/sell advice.
Explain which model performs better and why, based on the table.
Clarify that the comparison is based on the model testing period, not on live price movement at the current second.
Use short, neat sections.

Coin: {result["coin_name"]}
Comparison table: {json.dumps(payload, ensure_ascii=False, default=str)}
Best row: {json.dumps(best, ensure_ascii=False, default=str)}

Use exactly this format:
## 1. Best Performing Model
Write 1 paragraph explaining which model is stronger overall and which metrics support that conclusion.

## 2. Error Comparison
Write 1 paragraph explaining RMSE, MAE, and MAPE in simple terms. Lower error means the prediction is closer to the actual price.

## 3. Direction and Return Comparison
Write 1-2 paragraphs explaining Directional Accuracy and Cumulative Return in simple language.
Explain why Directional Accuracy can be moderate or low when the market moves unevenly, reverses often, or the model predicts the price level better than the daily direction.
Explain why Cumulative Return can be negative when the selected coin experienced a significant downtrend during the testing period. Make clear that negative return does not automatically mean the model is broken; it may reflect that the testing period itself was bearish, while the model's predicted path also followed or partially followed that weakness.
Do not say the result is influenced by a "real-time model"; say the comparison uses historical testing-period predictions, while live market data is only a separate dashboard context unless explicitly used in the model calculation.

## 4. Practical Meaning
Write 1 paragraph explaining what a general user should understand. Mention that crypto markets are volatile and both models are limited to historical data, statistical/machine-learning patterns, and available external variables.
"""

    return f"""
Tulis insight perbandingan model dalam Bahasa Indonesia yang mudah dipahami pengguna umum.
Jangan gunakan sapaan atau filler percakapan. Jangan memberi saran beli/jual.
Jelaskan model mana yang lebih baik dan alasannya berdasarkan tabel.
Jelaskan bahwa perbandingan ini berdasarkan periode testing model, bukan karena pergerakan harga live pada detik ini.
Gunakan section yang rapi dan singkat.

Koin: {result["coin_name"]}
Tabel perbandingan: {json.dumps(payload, ensure_ascii=False, default=str)}
Baris terbaik: {json.dumps(best, ensure_ascii=False, default=str)}

Gunakan format persis ini:
## 1. Model dengan Performa Terbaik
Tulis 1 paragraf yang menjelaskan model mana yang lebih kuat secara keseluruhan dan metrik apa yang mendukung kesimpulan tersebut.

## 2. Perbandingan Error
Tulis 1 paragraf yang menjelaskan RMSE, MAE, dan MAPE dengan bahasa sederhana. Error yang lebih kecil berarti prediksi lebih dekat dengan harga aktual.

## 3. Perbandingan Arah dan Return
Tulis 1-2 paragraf yang menjelaskan Directional Accuracy dan Cumulative Return dengan bahasa sederhana.
Jelaskan mengapa Directional Accuracy bisa sedang atau rendah ketika pasar bergerak tidak stabil, sering berbalik arah, atau ketika model lebih kuat membaca level harga dibanding arah harian.
Jelaskan mengapa Cumulative Return bisa negatif ketika koin terpilih mengalami downtrend signifikan selama periode testing. Tegaskan bahwa return negatif tidak otomatis berarti model rusak; hasil itu bisa mencerminkan bahwa periode testing memang bearish, sementara jalur prediksi model ikut mengikuti atau sebagian mengikuti pelemahan tersebut.
Jangan mengatakan hasil ini dipengaruhi oleh "real-time model"; jelaskan bahwa perbandingan memakai prediksi pada periode testing historis, sedangkan data live market hanya menjadi konteks dashboard terpisah kecuali memang digunakan dalam perhitungan model.

## 4. Makna Praktis
Tulis 1 paragraf tentang apa yang perlu dipahami pengguna umum. Sebutkan bahwa pasar kripto sangat volatil dan kedua model terbatas pada data historis, pola statistik/machine learning, serta variabel eksternal yang tersedia.
"""


def render_comparison_ai_insight(result: dict[str, object] | None) -> None:
    if result is None:
        st.markdown(
            f"""
            <section class="ai-insight-card comparison-ai-card">
                <div class="ai-insight-title">
                    <span class="ai-insight-icon">💡</span>
                    <span>AI Insight</span>
                </div>
                <div class="ai-insight-divider"></div>
                <p>{t("comparison_ai_placeholder")}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    api_key = get_gemini_api_key()
    model_name = get_gemini_model_name()
    if not api_key:
        st.markdown(
            f"""
            <section class="ai-insight-card comparison-ai-card">
                <div class="ai-insight-title">
                    <span class="ai-insight-icon">💡</span>
                    <span>AI Insight</span>
                </div>
                <div class="ai-insight-divider"></div>
                <p>{t("ai_missing")}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner(t("comparison_ai_loading")):
        prompt = build_comparison_ai_prompt(result)
        insight = request_gemini_insight(prompt, api_key, model_name, f"{AI_INSIGHT_CACHE_VERSION}-comparison")

    with st.container(key="comparison_ai_insight_card"):
        st.markdown(
            (
                '<div class="ai-insight-title">'
                '<span class="ai-insight-icon">💡</span>'
                "<span>AI Insight</span>"
                "</div>"
                '<div class="ai-insight-divider"></div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="ai-insight-scrollbox">{insight_text_to_html(insight)}</div>',
            unsafe_allow_html=True,
        )


def ensure_defaults() -> None:
    st.session_state.setdefault("coin_choice", None)
    st.session_state.setdefault("model_choice", None)
    st.session_state.setdefault("forecast_horizon_choice", None)
    st.session_state.setdefault("last_coin_choice", None)
    st.session_state.setdefault("last_model_choice", None)
    st.session_state.setdefault("last_forecast_horizon_choice", None)
    st.session_state.setdefault("prediction_result", None)

    if st.session_state.coin_choice is None and st.session_state.last_coin_choice is not None:
        st.session_state.coin_choice = st.session_state.last_coin_choice
    if st.session_state.model_choice is None and st.session_state.last_model_choice is not None:
        st.session_state.model_choice = st.session_state.last_model_choice
    if st.session_state.forecast_horizon_choice is None and st.session_state.last_forecast_horizon_choice is not None:
        st.session_state.forecast_horizon_choice = st.session_state.last_forecast_horizon_choice


@st.cache_data(ttl=3600, show_spinner=False)
def get_dataset_frame() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATA_PATH)


def describe_variable(column: str) -> tuple[str, str, str]:
    if column == "Date":
        return "Waktu", "Tanggal observasi harian untuk seluruh data kripto.", "Indeks waktu"
    if column.endswith("_price"):
        coin = column.replace("_price", "").title()
        return "Harga", f"Harga penutupan harian {coin} dalam USD.", "Target / referensi harga"
    if column.endswith("USDT_volume"):
        coin = column.replace("USDT_volume", "")
        return "Volume", f"Volume transaksi harian pasangan {coin}/USDT dari Binance.", "Fitur eksternal"
    if column.endswith("_ret_log"):
        coin = column.replace("USDT_ret_log", "")
        return "Return Log", f"Return log harian {coin}/USDT.", "Analisis return"
    if column == "FearGreed":
        return "Sentimen", "Nilai Fear & Greed Index harian sebagai proksi sentimen pasar.", "Fitur eksternal"
    if column == "FearGreed_norm":
        return "Sentimen", "Fear & Greed Index yang dinormalisasi ke rentang 0-1.", "Fitur rekayasa"
    if column.endswith("_Return"):
        coin = column.replace("_Return", "")
        return "Return", f"Return harian {coin} berbasis perubahan harga.", "Target XGBoost / fitur lag"
    if "Volatility_Monthly" in column:
        label = "market" if column == "Market_Volatility_Monthly" else column.split("_", 1)[0]
        return "Volatilitas", f"Volatilitas bulanan {label} berbasis rolling return 30 hari.", "Fitur risiko"
    if column == "Market_Volume":
        return "Market", "Total volume pasar dari BTC, ETH, dan SOL.", "Fitur eksternal"
    if "_MA_" in column:
        coin, window = column.split("_MA_")
        return "Moving Average", f"Rata-rata bergerak harga {coin} selama {window} hari.", "Fitur tren"
    if "_Vol_" in column:
        coin, window = column.split("_Vol_")
        return "Volatilitas", f"Volatilitas rolling return {coin} selama {window} hari.", "Fitur risiko"
    if "_Return_Lag_" in column:
        coin, lag = column.split("_Return_Lag_")
        return "Lag Return", f"Return {coin} pada {lag} hari sebelumnya.", "Fitur XGBoost"
    if "_Lag_" in column:
        coin, lag = column.split("_Lag_")
        return "Lag Harga", f"Harga {coin} pada {lag} hari sebelumnya.", "Fitur XGBoost"
    return "Lainnya", "Variabel turunan dari proses preprocessing dataset.", "Fitur pendukung"


def build_variable_catalog(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in dataset.columns:
        category, description, role = describe_variable(column)
        rows.append(
            {
                t("var_column"): column,
                t("category_column"): category,
                t("desc_column"): description,
                t("role_column"): role,
            }
        )
    return pd.DataFrame(rows)


def format_live_price(price: object) -> str:
    if price is None or pd.isna(price):
        return "-"
    return f"${float(price):,.2f}"


def change_class(change: object) -> str:
    if change is None or pd.isna(change):
        return "change-neutral"
    value = float(change)
    if value > 0.05:
        return "change-positive"
    if value < -0.05:
        return "change-negative"
    return "change-neutral"


def format_change(change: object) -> str:
    if change is None or pd.isna(change):
        return "-"
    value = float(change)
    prefix = "+" if value > 0 else ""
    return f"{prefix}{value:.2f}%"


@st.cache_data(ttl=60, show_spinner=False)
def initial_live_market_rows() -> list[dict[str, float | str | None]]:
    try:
        dataset = get_dataset_frame()
        latest = dataset.sort_values("Date").iloc[-1]
        return [
            {
                "ticker": "BTC",
                "price": float(latest.get("bitcoin_price", latest.get("BTC_Lag_1", 0))),
                "change": float(latest.get("BTC_Return", latest.get("BTC_Return_Lag_1", 0))) * 100,
            },
            {
                "ticker": "ETH",
                "price": float(latest.get("ethereum_price", latest.get("ETH_Lag_1", 0))),
                "change": float(latest.get("ETH_Return", latest.get("ETH_Return_Lag_1", 0))) * 100,
            },
            {
                "ticker": "SOL",
                "price": float(latest.get("solana_price", latest.get("SOL_Lag_1", 0))),
                "change": float(latest.get("SOL_Return", latest.get("SOL_Return_Lag_1", 0))) * 100,
            },
        ]
    except (IndexError, KeyError, TypeError, ValueError):
        return [
            {"ticker": "BTC", "price": None, "change": None},
            {"ticker": "ETH", "price": None, "change": None},
            {"ticker": "SOL", "price": None, "change": None},
        ]


def render_live_market_overview() -> None:
    initial_fear_greed = None
    initial_market_rows = initial_live_market_rows()
    labels = {
        "title": t("live_market_title"),
        "coin": "Coin",
        "price": t("live_price"),
        "change": t("change_24h"),
        "signal": t("market_signal"),
        "fear": t("fear_greed_title"),
        "unavailable": t("live_unavailable"),
        "statusReady": t("live_status_ready"),
        "statusSuccess": t("live_status_success"),
        "statusFailed": t("live_status_failed"),
    }
    component_html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: transparent; color: #F8FAFC; }}
            .live-market-section {{
                background: radial-gradient(circle at 8% 0%, rgba(56, 217, 169, 0.11), transparent 30%), #263545;
                border: 1px solid #425168;
                border-radius: 14px;
                padding: 1.15rem;
                box-shadow: 0 6px 22px rgba(5, 10, 18, 0.18);
            }}
            .live-section-header {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.95rem; }}
            .live-section-header h3 {{ margin: 0; color: #F8FAFC; font-size: 1.18rem; }}
            .live-status {{ margin-top: 0.32rem; color: #C0CAD8; font-size: 0.76rem; font-weight: 700; }}
            .live-status.ok {{ color: #38D9A9; }}
            .live-status.fail {{ color: #FF8A7A; }}
            .live-dot {{ width: 12px; height: 12px; border-radius: 999px; background: #38D9A9; box-shadow: 0 0 0 6px rgba(56, 217, 169, 0.12), 0 0 18px rgba(56, 217, 169, 0.42); animation: liveDotPulse 1.8s ease-in-out infinite; }}
            .live-market-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 1rem; align-items: stretch; }}
            .live-market-table-card, .fear-greed-card {{ background: #182231; border: 1px solid #425168; border-radius: 12px; overflow: hidden; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 0.85rem 0.95rem; border-bottom: 1px solid rgba(66, 81, 104, 0.65); text-align: left; vertical-align: middle; }}
            th {{ color: #C0CAD8; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; background: rgba(49, 66, 87, 0.5); }}
            tr:last-child td {{ border-bottom: 0; }}
            .live-coin-cell {{ display: flex; align-items: center; gap: 0.75rem; }}
            .coin-logo {{ width: 38px; height: 38px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center; color: #182231; font-weight: 900; font-size: 1.2rem; box-shadow: 0 0 16px rgba(5, 10, 18, 0.24); }}
            .coin-logo svg {{ width: 24px; height: 24px; display: block; }}
            .coin-btc {{ background: linear-gradient(135deg, #F7931A, #FFD166); }}
            .coin-eth {{ background: linear-gradient(135deg, #627EEA, #B7C5FF); }}
            .coin-sol {{ background: #101827; }}
            .live-coin-cell strong {{ display: block; color: #F8FAFC; font-size: 0.98rem; }}
            .live-coin-cell small {{ display: block; color: #C0CAD8; font-size: 0.78rem; font-weight: 700; }}
            .live-price {{ color: #F8FAFC; font-family: Courier New, monospace; font-weight: 850; transition: color 0.2s ease, text-shadow 0.2s ease; }}
            .price-flash-up {{ color: #38D9A9 !important; text-shadow: 0 0 14px rgba(56, 217, 169, 0.7); }}
            .price-flash-down {{ color: #FF6B7A !important; text-shadow: 0 0 14px rgba(255, 107, 122, 0.62); }}
            .change-pill, .signal-pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 86px; border-radius: 999px; padding: 0.34rem 0.62rem; font-size: 0.82rem; font-weight: 850; }}
            .positive, .bullish {{ color: #38D9A9; background: rgba(56, 217, 169, 0.12); border: 1px solid rgba(56, 217, 169, 0.38); }}
            .negative, .bearish {{ color: #FF6B7A; background: rgba(255, 107, 122, 0.12); border: 1px solid rgba(255, 107, 122, 0.36); }}
            .flat, .neutral {{ color: #C0CAD8; background: rgba(192, 202, 216, 0.1); border: 1px solid rgba(192, 202, 216, 0.28); }}
            .fear-greed-card {{ padding: 1rem; display: flex; flex-direction: column; justify-content: center; align-items: stretch; }}
            .fear-greed-title {{ color: #C0CAD8; font-size: 0.8rem; font-weight: 850; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.8rem; text-align: center; }}
            .fear-extreme, .fear {{ color: #FF6B7A; background: rgba(255, 107, 122, 0.14); }}
            .fg-neutral {{ color: #FFD166; background: rgba(255, 209, 102, 0.14); }}
            .greed, .greed-extreme {{ color: #38D9A9; background: rgba(56, 217, 169, 0.14); }}
            .fear-gauge {{ position: relative; width: min(100%, 242px); height: 172px; margin: 0.1rem auto 0; }}
            .fear-gauge-svg {{ width: 100%; height: 138px; display: block; overflow: visible; filter: drop-shadow(0 12px 20px rgba(5, 10, 18, 0.2)); }}
            .fear-gauge-svg path {{ fill: none; stroke-width: 17; stroke-linecap: butt; }}
            .fear-gauge-track {{ stroke: rgba(49, 66, 87, 0.42); }}
            .fear-gauge-fill {{ stroke: url(#fearGaugeGradient); }}
            .fear-gauge-marker {{ position: absolute; left: 15.4%; top: 71.5%; width: 20px; height: 20px; border-radius: 999px; transform: translate(-50%, -50%); background: #F8FAFC; border: 3px solid #F8FAFC; box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.08), 0 0 18px rgba(255, 255, 255, 0.36); transition: left 0.55s cubic-bezier(0.2, 0.9, 0.2, 1), top 0.55s cubic-bezier(0.2, 0.9, 0.2, 1); z-index: 5; }}
            .fear-gauge-score {{ position: absolute; left: 50%; top: 48%; transform: translate(-50%, -50%); z-index: 4; color: #F8FAFC; font-family: Inter, Segoe UI, Arial, sans-serif; font-size: 3.05rem; line-height: 1; font-weight: 900; letter-spacing: -0.04em; text-shadow: 0 10px 26px rgba(5, 10, 18, 0.28); }}
            .fear-greed-label {{ position: absolute; left: 50%; top: 78%; transform: translateX(-50%); z-index: 4; display: inline-flex; align-items: center; justify-content: center; width: max-content; max-width: calc(100% - 1.5rem); white-space: nowrap; border-radius: 999px; padding: 0.36rem 0.76rem; font-weight: 850; font-size: 0.94rem; line-height: 1; text-align: center; border: 1px solid rgba(192, 202, 216, 0.28); color: #C0CAD8; background: rgba(192, 202, 216, 0.1); }}
            .fear-greed-label.fear-extreme,
            .fear-greed-label.fear {{ color: #FF7A70; background: rgba(255, 122, 112, 0.13); border-color: rgba(255, 122, 112, 0.44); }}
            .fear-greed-label.fg-neutral {{ color: #FFE04D; background: rgba(255, 224, 77, 0.13); border-color: rgba(255, 224, 77, 0.44); }}
            .fear-greed-label.greed,
            .fear-greed-label.greed-extreme {{ color: #38D9A9; background: rgba(56, 217, 169, 0.13); border-color: rgba(56, 217, 169, 0.44); }}
            .fear-greed-card p {{ color: #C0CAD8; font-size: 0.82rem; font-weight: 650; margin: 0; }}
            @keyframes liveDotPulse {{ 0%, 100% {{ transform: scale(1); opacity: 0.75; }} 50% {{ transform: scale(1.16); opacity: 1; }} }}
            @media (max-width: 900px) {{ .live-market-grid {{ grid-template-columns: 1fr; }} }}
            @media (max-width: 640px) {{
                .live-market-section {{ padding: 0.85rem; border-radius: 12px; }}
                .live-section-header h3 {{ font-size: 1rem; }}
                th, td {{ padding: 0.68rem 0.55rem; }}
                th {{ font-size: 0.66rem; letter-spacing: 0.05em; }}
                .coin-logo {{ width: 32px; height: 32px; font-size: 1rem; }}
                .live-coin-cell {{ gap: 0.55rem; }}
                .live-coin-cell strong {{ font-size: 0.86rem; }}
                .live-coin-cell small {{ font-size: 0.7rem; }}
                .live-price {{ font-size: 0.82rem; }}
                .change-pill, .signal-pill {{ min-width: 68px; padding: 0.3rem 0.44rem; font-size: 0.72rem; }}
                .fear-greed-card {{ padding: 0.85rem; }}
                .fear-gauge {{ width: min(100%, 210px); height: 154px; }}
                .fear-gauge-svg {{ height: 122px; }}
                .fear-gauge-score {{ font-size: 2.45rem; }}
                .fear-greed-label {{ font-size: 0.86rem; }}
            }}
        </style>
    </head>
    <body>
        <section class="live-market-section">
            <div class="live-section-header">
                <div><h3 id="title"></h3><div class="live-status" id="liveStatus"></div></div><span class="live-dot"></span>
            </div>
            <div class="live-market-grid">
                <div class="live-market-table-card">
                    <table>
                        <thead><tr><th id="coinHead"></th><th id="priceHead"></th><th id="changeHead"></th><th id="signalHead"></th></tr></thead>
                        <tbody id="marketRows"></tbody>
                    </table>
                </div>
                <aside class="fear-greed-card">
                    <div class="fear-greed-title" id="fearTitle"></div>
                    <div class="fear-gauge" id="fgGauge">
                        <svg class="fear-gauge-svg" viewBox="0 0 260 170" aria-hidden="true">
                            <defs>
                                <linearGradient id="fearGaugeGradient" x1="38" y1="132" x2="222" y2="132" gradientUnits="userSpaceOnUse">
                                    <stop offset="0%" stop-color="#FF4E6A"></stop>
                                    <stop offset="27%" stop-color="#FF9F1A"></stop>
                                    <stop offset="50%" stop-color="#FFE04D"></stop>
                                    <stop offset="73%" stop-color="#99E600"></stop>
                                    <stop offset="100%" stop-color="#38D9A9"></stop>
                                </linearGradient>
                            </defs>
                            <path class="fear-gauge-track" d="M44 132 A86 86 0 0 1 216 132"></path>
                            <path class="fear-gauge-fill" d="M44 132 A86 86 0 0 1 216 132"></path>
                        </svg>
                        <div class="fear-gauge-marker" id="fgMarker"></div>
                        <div class="fear-gauge-score" id="fgScore">-</div>
                        <div class="fear-greed-label" id="fgLabel"></div>
                    </div>
                </aside>
            </div>
        </section>
        <script>
            const labels = {json.dumps(labels, ensure_ascii=False)};
            const initialFear = {json.dumps(initial_fear_greed or {}, ensure_ascii=False)};
            const initialMarket = {json.dumps(initial_market_rows, ensure_ascii=False)};
            const coins = [
                {{ name: 'Bitcoin', ticker: 'BTC', icon: '<span>₿</span>', logo: 'coin-btc', binance: 'BTCUSDT', gecko: 'bitcoin' }},
                {{ name: 'Ethereum', ticker: 'ETH', icon: '<svg viewBox="0 0 32 32" aria-hidden="true"><path fill="#FFFFFF" d="M16 2 7 16.4 16 21.8 25 16.4 16 2Z"/><path fill="#D8DEE9" d="M16 2v19.8l9-5.4L16 2Z"/><path fill="#FFFFFF" d="M7 18.2 16 30l9-11.8-9 5.3-9-5.3Z"/><path fill="#D8DEE9" d="M16 30V23.5l9-5.3L16 30Z"/></svg>', logo: 'coin-eth', binance: 'ETHUSDT', gecko: 'ethereum' }},
                {{ name: 'Solana', ticker: 'SOL', icon: '<svg viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="solGradient" x1="2" x2="30" y1="4" y2="28"><stop stop-color="#00FFA3"/><stop offset="0.5" stop-color="#03E1FF"/><stop offset="1" stop-color="#DC1FFF"/></linearGradient></defs><path fill="url(#solGradient)" d="M7.2 7.1c.3-.3.7-.5 1.1-.5h22c.7 0 1 .8.6 1.3l-4.1 4.1c-.3.3-.7.5-1.1.5h-22c-.7 0-1-.8-.6-1.3l4.1-4.1Zm0 19.8c.3-.3.7-.5 1.1-.5h22c.7 0 1 .8.6 1.3l-4.1 4.1c-.3.3-.7.5-1.1.5h-22c-.7 0-1-.8-.6-1.3l4.1-4.1Zm18.6-9.9c-.3-.3-.7-.5-1.1-.5h-22c-.7 0-1 .8-.6 1.3l4.1 4.1c.3.3.7.5 1.1.5h22c.7 0 1-.8.6-1.3L25.8 17Z"/></svg>', logo: 'coin-sol', binance: 'SOLUSDT', gecko: 'solana' }}
            ];
            const previousPrices = {{}};
            let priceSocket = null;
            let coinbaseSocket = null;
            let marketTimer = null;
            let reconnectTimer = null;
            let coinbaseReconnectTimer = null;
            const LIVE_REFRESH_MS = 5000;
            const API_TIMEOUT_MS = 2500;

            function byId(id) {{ return document.getElementById(id); }}
            function finiteOrNaN(value) {{ const parsed = Number(value); return Number.isFinite(parsed) ? parsed : NaN; }}
            function timeLabel() {{
                return new Date()
                    .toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }})
                    .replace(/^24:/, '00:');
            }}
            function setLiveStatus(message, state = '') {{
                const status = byId('liveStatus');
                if (!status) return;
                status.textContent = message + ' • ' + timeLabel();
                status.className = 'live-status ' + state;
            }}
            function fmtPrice(value) {{ return Number.isFinite(value) ? '$' + value.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}) : '-'; }}
            function fmtChange(value) {{ if (!Number.isFinite(value)) return '-'; return (value > 0 ? '+' : '') + value.toFixed(2) + '%'; }}
            function changeClass(value) {{ if (!Number.isFinite(value) || Math.abs(value) <= 0.05) return 'flat'; return value > 0 ? 'positive' : 'negative'; }}
            function signal(value) {{ if (!Number.isFinite(value)) return ['-', 'neutral']; if (value > 1) return ['Bullish', 'bullish']; if (value < -1) return ['Bearish', 'bearish']; return ['Neutral', 'neutral']; }}
            function fgBand(value) {{ if (value <= 24) return 'fear-extreme'; if (value <= 44) return 'fear'; if (value <= 55) return 'fg-neutral'; if (value <= 74) return 'greed'; return 'greed-extreme'; }}
            async function fetchNoStore(url, timeout = API_TIMEOUT_MS) {{
                const controller = new AbortController();
                const timer = window.setTimeout(() => controller.abort(), timeout);
                try {{
                    return await fetch(url, {{ cache: 'no-store', signal: controller.signal }});
                }} finally {{
                    window.clearTimeout(timer);
                }}
            }}

            function renderSkeleton() {{
                const initialByTicker = Object.fromEntries(initialMarket.map(row => [row.ticker, row]));
                byId('title').textContent = labels.title;
                setLiveStatus(labels.statusReady, '');
                byId('coinHead').textContent = labels.coin;
                byId('priceHead').textContent = labels.price;
                byId('changeHead').textContent = labels.change;
                byId('signalHead').textContent = labels.signal;
                byId('fearTitle').textContent = labels.fear;
                byId('fgLabel').textContent = labels.unavailable;
                byId('marketRows').innerHTML = coins.map(c => `
                    <tr>
                        <td><div class="live-coin-cell"><span class="coin-logo ${{c.logo}}">${{c.icon}}</span><div><strong>${{c.name}}</strong><small>${{c.ticker}}</small></div></div></td>
                        <td class="live-price" id="price-${{c.ticker}}">${{fmtPrice(finiteOrNaN(initialByTicker[c.ticker]?.price))}}</td>
                        <td><span class="change-pill ${{changeClass(finiteOrNaN(initialByTicker[c.ticker]?.change))}}" id="change-${{c.ticker}}">${{fmtChange(finiteOrNaN(initialByTicker[c.ticker]?.change))}}</span></td>
                        <td><span class="signal-pill ${{signal(finiteOrNaN(initialByTicker[c.ticker]?.change))[1]}}" id="signal-${{c.ticker}}">${{signal(finiteOrNaN(initialByTicker[c.ticker]?.change))[0]}}</span></td>
                    </tr>`).join('');
                initialMarket.forEach(row => {{
                    const price = finiteOrNaN(row.price);
                    if (Number.isFinite(price)) previousPrices[row.ticker] = price;
                }});
            }}

            function setFearGreed(item) {{
                if (!item || item.value === undefined || item.value === null) return;
                const value = Number(item.value);
                const band = item.band || fgBand(value);
                byId('fgScore').textContent = Number.isFinite(value) ? String(value) : '-';
                byId('fgScore').className = 'fear-gauge-score ' + band;
                byId('fgLabel').textContent = item.classification || labels.unavailable;
                byId('fgLabel').className = 'fear-greed-label ' + band;
                const marker = byId('fgMarker');
                if (marker) {{
                    const bounded = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 50;
                    const angle = (180 - bounded * 1.8) * Math.PI / 180;
                    const cx = 130;
                    const cy = 132;
                    const radius = 86;
                    const x = cx + radius * Math.cos(angle);
                    const y = cy - radius * Math.sin(angle);
                    marker.style.left = `${{(x / 260) * 100}}%`;
                    marker.style.top = `${{(y / 170) * 100}}%`;
                }}
            }}

            async function fetchBinance() {{
                const now = Date.now();
                const symbols = encodeURIComponent(JSON.stringify(coins.map(c => c.binance)));
                const response = await fetchNoStore(`https://api.binance.com/api/v3/ticker/24hr?symbols=${{symbols}}&_=${{now}}`);
                if (!response.ok) throw new Error('Binance unavailable');
                const data = await response.json();
                return coins.map(c => {{
                    const row = data.find(item => item.symbol === c.binance) || {{}};
                    return {{ ticker: c.ticker, price: Number(row.lastPrice), change: Number(row.priceChangePercent) }};
                }});
            }}

            async function fetchBinanceOneByOne() {{
                const now = Date.now();
                const rows = await Promise.all(coins.map(async c => {{
                    const response = await fetchNoStore(`https://api.binance.com/api/v3/ticker/24hr?symbol=${{c.binance}}&_=${{now}}`);
                    if (!response.ok) throw new Error('Binance single unavailable');
                    const row = await response.json();
                    return {{ ticker: c.ticker, price: Number(row.lastPrice), change: Number(row.priceChangePercent) }};
                }}));
                return rows;
            }}

            async function fetchCoinGecko() {{
                const now = Date.now();
                const ids = coins.map(c => c.gecko).join(',');
                const response = await fetchNoStore(`https://api.coingecko.com/api/v3/simple/price?ids=${{ids}}&vs_currencies=usd&include_24hr_change=true&_=${{now}}`);
                if (!response.ok) throw new Error('CoinGecko unavailable');
                const data = await response.json();
                return coins.map(c => ({{ ticker: c.ticker, price: Number(data[c.gecko]?.usd), change: Number(data[c.gecko]?.usd_24h_change) }}));
            }}

            async function fetchCoinCap() {{
                const now = Date.now();
                const response = await fetchNoStore(`https://api.coincap.io/v2/assets?ids=bitcoin,ethereum,solana&_=${{now}}`);
                if (!response.ok) throw new Error('CoinCap unavailable');
                const payload = await response.json();
                const data = payload.data || [];
                return coins.map(c => {{
                    const row = data.find(item => item.id === c.gecko) || {{}};
                    return {{ ticker: c.ticker, price: Number(row.priceUsd), change: Number(row.changePercent24Hr) }};
                }});
            }}

            async function fetchCryptoCompare() {{
                const now = Date.now();
                const response = await fetchNoStore(`https://min-api.cryptocompare.com/data/pricemultifull?fsyms=BTC,ETH,SOL&tsyms=USD&_=${{now}}`);
                if (!response.ok) throw new Error('CryptoCompare unavailable');
                const payload = await response.json();
                const raw = payload.RAW || {{}};
                return coins.map(c => {{
                    const row = raw[c.ticker]?.USD || {{}};
                    return {{ ticker: c.ticker, price: Number(row.PRICE), change: Number(row.CHANGEPCT24HOUR) }};
                }});
            }}

            async function fetchCoinbase() {{
                const pairs = {{ BTC: 'BTC-USD', ETH: 'ETH-USD', SOL: 'SOL-USD' }};
                const now = Date.now();
                const rows = await Promise.all(coins.map(async c => {{
                    const pair = pairs[c.ticker];
                    const response = await fetchNoStore(`https://api.exchange.coinbase.com/products/${{pair}}/stats?_=${{now}}`);
                    if (!response.ok) throw new Error('Coinbase unavailable');
                    const row = await response.json();
                    const price = Number(row.last);
                    const open = Number(row.open);
                    const change = Number.isFinite(price) && Number.isFinite(open) && open !== 0 ? ((price - open) / open) * 100 : NaN;
                    return {{ ticker: c.ticker, price, change }};
                }}));
                return rows;
            }}

            function applyTickerRow(row) {{
                const priceEl = byId('price-' + row.ticker);
                const changeEl = byId('change-' + row.ticker);
                const signalEl = byId('signal-' + row.ticker);
                if (!priceEl || !changeEl || !signalEl) return;
                const oldPrice = previousPrices[row.ticker];
                priceEl.textContent = fmtPrice(row.price);
                if (Number.isFinite(row.price) && oldPrice !== undefined && row.price !== oldPrice) {{
                    const flashClass = row.price > oldPrice ? 'price-flash-up' : 'price-flash-down';
                    priceEl.classList.remove('price-flash-up', 'price-flash-down');
                    void priceEl.offsetWidth;
                    priceEl.classList.add(flashClass);
                    setTimeout(() => priceEl.classList.remove(flashClass), 500);
                }}
                if (Number.isFinite(row.price)) previousPrices[row.ticker] = row.price;
                if (Number.isFinite(row.change)) {{
                    changeEl.textContent = fmtChange(row.change);
                    changeEl.className = 'change-pill ' + changeClass(row.change);
                    const [signalText, signalClass] = signal(row.change);
                    signalEl.textContent = signalText;
                    signalEl.className = 'signal-pill ' + signalClass;
                }}
            }}

            async function updateMarket() {{
                let rows;
                try {{
                    rows = await fetchBinance();
                }} catch (err) {{
                    try {{
                        rows = await fetchBinanceOneByOne();
                    }} catch (singleErr) {{
                        try {{
                            rows = await fetchCoinGecko();
                        }} catch (geckoErr) {{
                            try {{
                                rows = await fetchCoinCap();
                            }} catch (coincapErr) {{
                                try {{
                                    rows = await fetchCryptoCompare();
                                }} catch (cryptoCompareErr) {{
                                    try {{
                                        rows = await fetchCoinbase();
                                    }} catch (coinbaseErr) {{
                                        setLiveStatus(labels.statusFailed, 'fail');
                                        return;
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
                rows.forEach(applyTickerRow);
                setLiveStatus(labels.statusSuccess, 'ok');
            }}

            function startMarketPolling() {{
                if (marketTimer) return;
                updateMarket();
                marketTimer = window.setInterval(updateMarket, LIVE_REFRESH_MS);
            }}

            function openPriceStream() {{
                if (priceSocket && (priceSocket.readyState === WebSocket.OPEN || priceSocket.readyState === WebSocket.CONNECTING)) return;
                const streams = coins.map(c => c.binance.toLowerCase() + '@ticker').join('/');
                priceSocket = new WebSocket('wss://stream.binance.com/stream?streams=' + streams);

                priceSocket.onopen = () => {{
                    if (reconnectTimer) {{
                        window.clearTimeout(reconnectTimer);
                        reconnectTimer = null;
                    }}
                }};

                priceSocket.onmessage = event => {{
                    try {{
                        const payload = JSON.parse(event.data);
                        const data = payload.data || payload;
                        const coin = coins.find(c => c.binance === data.s);
                        if (!coin) return;
                        applyTickerRow({{ ticker: coin.ticker, price: Number(data.c), change: Number(data.P) }});
                    }} catch (err) {{}}
                }};

                priceSocket.onerror = () => {{
                    try {{ priceSocket.close(); }} catch (err) {{}}
                }};

                priceSocket.onclose = () => {{
                    if (!reconnectTimer) {{
                        reconnectTimer = window.setTimeout(() => {{
                            reconnectTimer = null;
                            openPriceStream();
                        }}, 3000);
                    }}
                }};
            }}

            function openCoinbaseStream() {{
                if (coinbaseSocket && (coinbaseSocket.readyState === WebSocket.OPEN || coinbaseSocket.readyState === WebSocket.CONNECTING)) return;
                const pairs = {{ BTC: 'BTC-USD', ETH: 'ETH-USD', SOL: 'SOL-USD' }};
                const tickerByPair = Object.fromEntries(Object.entries(pairs).map(([ticker, pair]) => [pair, ticker]));
                coinbaseSocket = new WebSocket('wss://ws-feed.exchange.coinbase.com');

                coinbaseSocket.onopen = () => {{
                    coinbaseSocket.send(JSON.stringify({{
                        type: 'subscribe',
                        product_ids: Object.values(pairs),
                        channels: ['ticker']
                    }}));
                    if (coinbaseReconnectTimer) {{
                        window.clearTimeout(coinbaseReconnectTimer);
                        coinbaseReconnectTimer = null;
                    }}
                }};

                coinbaseSocket.onmessage = event => {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.type !== 'ticker') return;
                        const ticker = tickerByPair[data.product_id];
                        if (!ticker) return;
                        applyTickerRow({{ ticker, price: Number(data.price), change: NaN }});
                    }} catch (err) {{}}
                }};

                coinbaseSocket.onerror = () => {{
                    try {{ coinbaseSocket.close(); }} catch (err) {{}}
                }};

                coinbaseSocket.onclose = () => {{
                    if (!coinbaseReconnectTimer) {{
                        coinbaseReconnectTimer = window.setTimeout(() => {{
                            coinbaseReconnectTimer = null;
                            openCoinbaseStream();
                        }}, 3000);
                    }}
                }};
            }}

            async function updateFearGreed() {{
                try {{
                    const response = await fetchNoStore('https://api.alternative.me/fng/?limit=1', 3500);
                    if (!response.ok) throw new Error('Fear Greed unavailable');
                    const payload = await response.json();
                    const item = payload.data?.[0];
                    if (item) setFearGreed({{ value: Number(item.value), classification: item.value_classification, band: fgBand(Number(item.value)) }});
                }} catch (err) {{ setFearGreed(initialFear); }}
            }}

            renderSkeleton();
            setFearGreed(initialFear);
            startMarketPolling();
            openPriceStream();
            openCoinbaseStream();
            updateFearGreed();
            window.setInterval(updateFearGreed, 30000);
        </script>
    </body>
    </html>
    """
    components.html(component_html, height=360, scrolling=True)
    st.markdown('<div class="live-market-bottom-space"></div>', unsafe_allow_html=True)


def clear_prediction_controls() -> None:
    st.session_state.coin_choice = None
    st.session_state.model_choice = None
    st.session_state.forecast_horizon_choice = None
    st.session_state.last_coin_choice = None
    st.session_state.last_model_choice = None
    st.session_state.last_forecast_horizon_choice = None
    st.session_state.prediction_result = None


def render_controls() -> tuple[str | None, str | None, int, bool]:
    ensure_defaults()
    with st.container(key="control_panel"):
        input_cols = st.columns(3, gap="large", vertical_alignment="top")
        with input_cols[0]:
            coin_display = st.selectbox(
                t("choose_coin"),
                coin_display_options(),
                index=None,
                placeholder=t("coin_placeholder"),
                key="coin_choice",
            )
            st.markdown(f"<div class='control-help'>{t('coin_options')}</div>", unsafe_allow_html=True)

        with input_cols[1]:
            model_name = st.selectbox(
                t("choose_model"),
                list(MODEL_OPTIONS.keys()),
                index=None,
                placeholder=t("model_placeholder"),
                key="model_choice",
            )
            st.markdown(f"<div class='control-help'>{t('model_options')}</div>", unsafe_allow_html=True)

        with input_cols[2]:
            horizon = st.selectbox(
                t("choose_horizon"),
                horizon_options(),
                index=None,
                placeholder=t("horizon_placeholder"),
                key="forecast_horizon_choice",
            )
            st.markdown(f"<div class='control-help'>{t('horizon_options')}</div>", unsafe_allow_html=True)

        st.markdown('<div class="backtest-control-button-row"></div>', unsafe_allow_html=True)
        button_cols = st.columns(3, gap="large", vertical_alignment="top")
        with button_cols[1]:
            st.markdown(f"<div class='button-label-spacer'>{t('action')}</div>", unsafe_allow_html=True)
            st.button(t("reset_date"), key="prediction_reset_button", use_container_width=True, on_click=clear_prediction_controls)
        with button_cols[2]:
            st.markdown(f"<div class='button-label-spacer'>{t('action')}</div>", unsafe_allow_html=True)
            clicked = st.button(t("start_prediction"), use_container_width=True)

    coin_name = coin_name_from_display(coin_display) if coin_display else None
    return coin_name, model_name, horizon_days(horizon), clicked


def render_prediction_page() -> None:
    render_header()
    render_live_market_overview()
    coin_name, model_name, selected_days, clicked = render_controls()

    if (
        st.session_state.prediction_result is not None
        and selected_days
        and st.session_state.prediction_result.get("forecast_days") != selected_days
    ):
        st.session_state.prediction_result = None

    if clicked:
        if not coin_name or not model_name or not selected_days:
            st.warning(t("select_warning"))
        else:
            with st.spinner(t("loading_prediction")):
                st.session_state.prediction_result = run_prediction(
                    coin_name=coin_name,
                    model_name=model_name,
                    data_path=DEFAULT_DATA_PATH,
                    days=selected_days,
                )
                st.session_state.last_coin_choice = st.session_state.coin_choice
                st.session_state.last_model_choice = st.session_state.model_choice
                st.session_state.last_forecast_horizon_choice = st.session_state.forecast_horizon_choice

    result = st.session_state.prediction_result
    if result is not None and result.get("model_name") == "SARIMAX" and "residual_diagnostics" not in result:
        with st.spinner(t("loading_prediction")):
            st.session_state.prediction_result = run_prediction(
                coin_name=result["coin_name"],
                model_name=result["model_name"],
                data_path=DEFAULT_DATA_PATH,
                days=int(result.get("forecast_days", selected_days or 7)),
            )
            result = st.session_state.prediction_result

    st.markdown(f'<div class="section-title">{t("actual_vs_prediction")}</div>', unsafe_allow_html=True)
    if result is None:
        st.markdown(
            f'<div class="placeholder">{t("chart_placeholder")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chart-caption">{t("chart_caption")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="section-title">{t("metrics_title")}</div>', unsafe_allow_html=True)
        render_metric_grid(
            {
                "RMSE": float("nan"),
                "MAE": float("nan"),
                "MAPE": float("nan"),
                "R2": float("nan"),
                "Directional Accuracy": float("nan"),
                "Cumulative Return": float("nan"),
            }
        )
        st.markdown(f'<div class="section-title">{forecast_title(selected_days)}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="placeholder">{t("forecast_placeholder")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <section class="ai-insight-card">
                <div class="ai-insight-title">
                    <span class="ai-insight-icon">💡</span>
                    <span>AI Insight</span>
                </div>
                <div class="ai-insight-divider"></div>
                <p>
                    {t("ai_placeholder")}
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    chart_title = f"{result['coin_name']} - {t('actual_vs_prediction')} ({result['model_name']})"
    if result.get("data_source") == "local":
        st.warning(t("local_data_warning"))
    st.plotly_chart(prediction_chart(result["test_predictions"], chart_title), use_container_width=True)
    test_start, test_end = result["test_period"]
    st.markdown(
        (
            '<div class="chart-caption">'
            f'{t("chart_caption")} '
            f'{t("test_period")}: {format_period_date(test_start)} - {format_period_date(test_end)}.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="section-title">{t("metrics_title")}</div>', unsafe_allow_html=True)
    actual_mean = float(result["test_predictions"]["Actual Price"].mean())
    render_metric_grid(result["metrics"], actual_mean)

    if result["model_name"] == "XGBoost" and "feature_importance" in result:
        st.markdown(
            f'<div class="section-title" style="margin-top: 1.5rem;">{t("feature_title")}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            feature_importance_chart(
                result["feature_importance"],
                f"{result['coin_name']} - Top 10 Feature Importance (XGBoost)",
            ),
            use_container_width=True,
        )
        st.markdown(f'<div class="chart-caption">{t("feature_caption")}</div>', unsafe_allow_html=True)
    elif result["model_name"] == "SARIMAX":
        render_sarimax_diagnostics(result)

    result_days = int(result.get("forecast_days", selected_days))
    st.markdown(f'<div class="section-title" style="margin-top: 1.5rem;">{forecast_title(result_days)}</div>', unsafe_allow_html=True)
    st.plotly_chart(
        forecast_chart(
            result["forecast_chart"],
            f"{result['coin_name']} - {forecast_title(result_days)} ({result['model_name']})",
        ),
        use_container_width=True,
    )
    render_prediction_confidence_cards(result)
    render_ai_insight(result)


def clear_comparison_controls() -> None:
    st.session_state.comparison_coin_choice = None
    st.session_state.last_comparison_coin_choice = None
    st.session_state.comparison_result = None


def render_comparison_controls() -> tuple[str | None, bool]:
    st.session_state.setdefault("comparison_coin_choice", None)
    st.session_state.setdefault("last_comparison_coin_choice", None)
    st.session_state.setdefault("comparison_result", None)
    if st.session_state.comparison_coin_choice is None and st.session_state.last_comparison_coin_choice is not None:
        st.session_state.comparison_coin_choice = st.session_state.last_comparison_coin_choice

    with st.container(key="control_panel"):
        control_cols = st.columns([1.18, 0.72, 0.46], gap="large", vertical_alignment="top")
        with control_cols[0]:
            coin_display = st.selectbox(
                t("choose_coin"),
                coin_display_options(),
                index=None,
                placeholder=t("coin_placeholder"),
                key="comparison_coin_choice",
            )
            st.markdown(f"<div class='control-help'>{t('coin_options')}</div>", unsafe_allow_html=True)
        with control_cols[1]:
            st.markdown(f"<div class='button-label-spacer'>{t('action')}</div>", unsafe_allow_html=True)
            clicked = st.button(t("comparison_start"), use_container_width=True)
        with control_cols[2]:
            st.markdown(f"<div class='button-label-spacer'>{t('action')}</div>", unsafe_allow_html=True)
            st.button(t("reset_date"), key="comparison_reset_button", use_container_width=True, on_click=clear_comparison_controls)

    coin_name = coin_name_from_display(coin_display) if coin_display else None
    return coin_name, clicked


def render_comparison_placeholders() -> None:
    st.markdown('<div class="comparison-section-spacer"></div>', unsafe_allow_html=True)
    first_row = st.columns(2, gap="large")
    with first_row[0]:
        with st.container(key="comparison_table_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_table_title")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="placeholder comparison-placeholder">{t("comparison_table_placeholder")}</div>', unsafe_allow_html=True)
    with first_row[1]:
        with st.container(key="comparison_error_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_error_title")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="placeholder comparison-placeholder">{t("comparison_error_placeholder")}</div>', unsafe_allow_html=True)

    second_row = st.columns(2, gap="large")
    with second_row[0]:
        with st.container(key="comparison_direction_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_direction_title")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="placeholder comparison-placeholder">{t("comparison_direction_placeholder")}</div>', unsafe_allow_html=True)
    with second_row[1]:
        with st.container(key="comparison_return_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_return_title")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="placeholder comparison-placeholder">{t("comparison_return_placeholder")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="comparison-ai-spacer"></div>', unsafe_allow_html=True)
    render_comparison_ai_insight(None)


def render_comparison_page() -> None:
    render_header(title=t("comparison_title"), subtitle=t("comparison_subtitle"))
    top_cols = st.columns([1.25, 0.9], gap="large", vertical_alignment="top")
    with top_cols[0]:
        coin_name, clicked = render_comparison_controls()

    if clicked:
        if not coin_name:
            st.warning(t("comparison_warning"))
        else:
            with st.spinner(t("comparison_loading")):
                st.session_state.comparison_result = run_model_comparison(
                    coin_name=coin_name,
                    data_path=str(DEFAULT_DATA_PATH),
                )
                st.session_state.last_comparison_coin_choice = st.session_state.comparison_coin_choice

    result = st.session_state.get("comparison_result")
    if result is not None and coin_name and result.get("coin_name") != coin_name:
        st.session_state.comparison_result = None
        result = None

    with top_cols[1]:
        render_comparison_best_cards(result)

    if result is None:
        render_comparison_placeholders()
        return

    table = result["comparison_table"]
    st.markdown('<div class="comparison-section-spacer"></div>', unsafe_allow_html=True)
    first_row = st.columns(2, gap="large")
    with first_row[0]:
        with st.container(key="comparison_table_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_table_title")}</div>', unsafe_allow_html=True)
            st.dataframe(format_comparison_table(table), use_container_width=True, hide_index=True, height=180)
    with first_row[1]:
        with st.container(key="comparison_error_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_error_title")}</div>', unsafe_allow_html=True)
            st.plotly_chart(comparison_error_chart(table), use_container_width=True)

    second_row = st.columns(2, gap="large")
    with second_row[0]:
        with st.container(key="comparison_direction_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_direction_title")}</div>', unsafe_allow_html=True)
            st.plotly_chart(comparison_direction_chart(table), use_container_width=True)
    with second_row[1]:
        with st.container(key="comparison_return_card"):
            st.markdown(f'<div class="comparison-card-title">{t("comparison_return_title")}</div>', unsafe_allow_html=True)
            st.plotly_chart(comparison_return_chart(table), use_container_width=True)

    st.markdown('<div class="comparison-ai-spacer"></div>', unsafe_allow_html=True)
    render_comparison_ai_insight(result)


def clear_backtesting_controls() -> None:
    st.session_state.backtest_coin_choice = None
    st.session_state.backtest_model_choice = None
    st.session_state.backtest_horizon_choice = None
    st.session_state.backtest_date_choice = None
    st.session_state.last_backtest_coin_choice = None
    st.session_state.last_backtest_model_choice = None
    st.session_state.last_backtest_horizon_choice = None
    st.session_state.last_backtest_date_choice = None
    st.session_state.backtesting_result = None


def render_backtesting_controls() -> tuple[str | None, str | None, pd.Timestamp | None, int, bool]:
    dataset = get_dataset_frame()
    min_date = pd.to_datetime(dataset["Date"]).min().date()
    st.session_state.setdefault("backtest_coin_choice", None)
    st.session_state.setdefault("backtest_model_choice", None)
    st.session_state.setdefault("backtest_horizon_choice", None)
    st.session_state.setdefault("backtest_date_choice", None)
    st.session_state.setdefault("last_backtest_coin_choice", None)
    st.session_state.setdefault("last_backtest_model_choice", None)
    st.session_state.setdefault("last_backtest_horizon_choice", None)
    st.session_state.setdefault("last_backtest_date_choice", None)

    if st.session_state.backtest_coin_choice is None and st.session_state.last_backtest_coin_choice is not None:
        st.session_state.backtest_coin_choice = st.session_state.last_backtest_coin_choice
    if st.session_state.backtest_model_choice is None and st.session_state.last_backtest_model_choice is not None:
        st.session_state.backtest_model_choice = st.session_state.last_backtest_model_choice
    if st.session_state.backtest_horizon_choice is None and st.session_state.last_backtest_horizon_choice is not None:
        st.session_state.backtest_horizon_choice = st.session_state.last_backtest_horizon_choice
    if st.session_state.backtest_date_choice is None and st.session_state.last_backtest_date_choice is not None:
        st.session_state.backtest_date_choice = st.session_state.last_backtest_date_choice

    selected_days = horizon_days(st.session_state.get("backtest_horizon_choice"))
    date_offset_days = selected_days or 7
    max_date = (pd.to_datetime(dataset["Date"]).max() - pd.Timedelta(days=date_offset_days)).date()
    if st.session_state.backtest_date_choice and st.session_state.backtest_date_choice > max_date:
        st.session_state.backtest_date_choice = None

    with st.container(key="control_panel"):
        input_cols = st.columns([1.05, 1.05, 0.9, 1.05], gap="large", vertical_alignment="top")
        with input_cols[0]:
            coin_display = st.selectbox(
                t("choose_coin"),
                coin_display_options(),
                index=None,
                placeholder=t("coin_placeholder"),
                key="backtest_coin_choice",
            )
            st.markdown(f"<div class='control-help'>{t('coin_options')}</div>", unsafe_allow_html=True)
        with input_cols[1]:
            model_name = st.selectbox(
                t("choose_model"),
                list(MODEL_OPTIONS.keys()),
                index=None,
                placeholder=t("model_placeholder"),
                key="backtest_model_choice",
            )
            st.markdown(f"<div class='control-help'>{t('model_options')}</div>", unsafe_allow_html=True)
        with input_cols[2]:
            horizon = st.selectbox(
                t("choose_horizon"),
                horizon_options(),
                index=None,
                placeholder=t("horizon_placeholder"),
                key="backtest_horizon_choice",
            )
            selected_days = horizon_days(horizon)
            st.markdown(f"<div class='control-help'>{t('horizon_options')}</div>", unsafe_allow_html=True)
        with input_cols[3]:
            selected_date = st.date_input(
                t("choose_date"),
                value=None,
                min_value=min_date,
                max_value=max_date,
                format="YYYY/MM/DD",
                key="backtest_date_choice",
            )
            st.markdown(f"<div class='control-help'>{t('date_options')}</div>", unsafe_allow_html=True)

        st.markdown('<div class="backtest-control-button-row"></div>', unsafe_allow_html=True)
        button_cols = st.columns([1.05, 1.05, 0.9, 1.05], gap="large", vertical_alignment="top")
        with button_cols[2]:
            st.markdown(f"<div class='button-label-spacer'>{t('action')}</div>", unsafe_allow_html=True)
            st.button(t("reset_date"), key="backtesting_reset_button", use_container_width=True, on_click=clear_backtesting_controls)
        with button_cols[3]:
            st.markdown(f"<div class='button-label-spacer'>{t('action')}</div>", unsafe_allow_html=True)
            clicked = st.button(t("start_backtest"), use_container_width=True)

    coin_name = coin_name_from_display(coin_display) if coin_display else None
    return coin_name, model_name, pd.Timestamp(selected_date) if selected_date else None, selected_days, clicked


def confidence_card_class(level: str) -> str:
    if level == "High":
        return "metric-good"
    if level == "Medium":
        return "metric-medium"
    return "metric-bad"


def render_backtest_summary_cards(result: dict[str, object]) -> None:
    start, end = result["backtest_period"]
    result_days = int(result.get("forecast_days", 7))
    cards = [
        f"""
        <div class="metric-card {confidence_card_class(result["confidence_level"])}">
            <div class="metric-label">{t("prediction_reliability")}</div>
            <div class="metric-value">{result["confidence_level"]}</div>
            <div class="metric-help">{t("confidence_level")}</div>
            <div class="metric-tooltip">{html.escape(card_tooltip("prediction_reliability"))}</div>
        </div>
        """,
        f"""
        <div class="metric-card {confidence_card_class(result["confidence_level"])}">
            <div class="metric-label">{t("confidence_level")}</div>
            <div class="metric-value">{result["confidence_score"]:.2f}%</div>
            <div class="metric-help">High 80-100 | Medium 60-79 | Low &lt;60</div>
            <div class="metric-tooltip">{html.escape(card_tooltip("confidence_level"))}</div>
        </div>
        """,
        f"""
        <div class="metric-card">
            <div class="metric-label">{t("backtest_period")}</div>
            <div class="metric-value dataset-summary-value">{format_period_date(start)} - {format_period_date(end)}</div>
            <div class="metric-help">{result_days} days historical simulation</div>
            <div class="metric-tooltip">{html.escape(card_tooltip("backtest_period"))}</div>
        </div>
        """,
    ]
    render_card_grid(cards, "backtest-summary-grid")


def render_backtesting_page() -> None:
    render_header(title=t("backtesting_title"), subtitle=t("backtesting_subtitle"))
    coin_name, model_name, selected_date, selected_days, clicked = render_backtesting_controls()

    if (
        st.session_state.get("backtesting_result") is not None
        and selected_days
        and st.session_state.backtesting_result.get("forecast_days") != selected_days
    ):
        st.session_state.backtesting_result = None

    if clicked:
        if not coin_name or not model_name or selected_date is None or not selected_days:
            st.warning(t("backtest_select_warning"))
        else:
            with st.spinner(t("loading_backtest")):
                st.session_state.backtesting_result = run_backtesting(
                    coin_name=coin_name,
                    model_name=model_name,
                    cutoff_date=selected_date,
                    data_path=DEFAULT_DATA_PATH,
                    days=selected_days,
                )
                st.session_state.last_backtest_coin_choice = st.session_state.backtest_coin_choice
                st.session_state.last_backtest_model_choice = st.session_state.backtest_model_choice
                st.session_state.last_backtest_horizon_choice = st.session_state.backtest_horizon_choice
                st.session_state.last_backtest_date_choice = st.session_state.backtest_date_choice

    result = st.session_state.get("backtesting_result")
    chart_col, table_col = st.columns([1.18, 0.92], gap="large")
    with chart_col:
        st.markdown(f'<div class="section-title">{t("backtest_chart_title")}</div>', unsafe_allow_html=True)
        if result is None:
            st.markdown(f'<div class="placeholder backtest-placeholder">{t("backtest_chart_placeholder")}</div>', unsafe_allow_html=True)
        else:
            st.plotly_chart(
                backtest_chart(
                    result["chart_data"],
                    f"{result['coin_name']} - {t('backtest_chart_title')} ({result['model_name']})",
                    result["cutoff_date"],
                ),
                use_container_width=True,
            )

    with table_col:
        st.markdown(f'<div class="section-title">{t("backtest_table_title")}</div>', unsafe_allow_html=True)
        if result is None:
            st.markdown(f'<div class="placeholder backtest-placeholder">{t("backtest_table_placeholder")}</div>', unsafe_allow_html=True)
        else:
            table = result["comparison_table"].copy()
            table["Date"] = pd.to_datetime(table["Date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(table, use_container_width=True, hide_index=True, height=360)

    st.markdown('<div class="backtest-section-spacer"></div>', unsafe_allow_html=True)

    if result is None:
        render_card_grid(
            [
                f"""
                <div class="metric-card">
                    <div class="metric-label">{t("prediction_reliability")}</div>
                    <div class="metric-value">-</div>
                    <div class="metric-help">High / Medium / Low</div>
                    <div class="metric-tooltip">{html.escape(card_tooltip("prediction_reliability"))}</div>
                </div>
                """,
                f"""
                <div class="metric-card">
                    <div class="metric-label">{t("confidence_level")}</div>
                    <div class="metric-value">-</div>
                    <div class="metric-help">High 80-100 | Medium 60-79 | Low &lt;60</div>
                    <div class="metric-tooltip">{html.escape(card_tooltip("confidence_level"))}</div>
                </div>
                """,
                f"""
                <div class="metric-card">
                    <div class="metric-label">{t("backtest_period")}</div>
                    <div class="metric-value dataset-summary-value">-</div>
                    <div class="metric-help">{selected_days} days historical simulation</div>
                    <div class="metric-tooltip">{html.escape(card_tooltip("backtest_period"))}</div>
                </div>
                """,
            ],
            "backtest-summary-grid",
        )
        st.markdown(
            f"""
            <section class="ai-insight-card">
                <div class="ai-insight-title">
                    <span class="ai-insight-icon">💡</span>
                    <span>AI Insight</span>
                </div>
                <div class="ai-insight-divider"></div>
                <p>
                    {t("backtest_ai_placeholder")}
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    render_backtest_summary_cards(result)
    render_backtest_ai_insight(result)


def render_dataset_page() -> None:
    render_header(
        title=t("dataset_title"),
        subtitle=t("dataset_subtitle"),
    )

    with st.spinner(t("dataset_loading")):
        dataset = get_dataset_frame()

    source_label = t("live_source") if dataset.attrs.get("source") == "live" else t("local_source")
    date_min = pd.to_datetime(dataset["Date"]).min()
    date_max = pd.to_datetime(dataset["Date"]).max()

    st.markdown(f'<div class="section-title">{t("dataset_summary")}</div>', unsafe_allow_html=True)
    summary_items = [
        (t("data_source"), source_label, t("active_dataset_source")),
        (t("row_count"), f"{len(dataset):,}", t("total_rows")),
        (t("column_count"), f"{len(dataset.columns):,}", t("total_columns")),
        (t("date_range"), f"{format_period_date(date_min)} - {format_period_date(date_max)}", t("final_period")),
    ]

    summary_cards = [
        f"""
        <div class="metric-card dataset-summary-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value dataset-summary-value">{value}</div>
            <div class="metric-help">{helper}</div>
        </div>
        """
        for label, value, helper in summary_items
    ]
    render_card_grid(summary_cards, "dataset-summary-grid")

    st.markdown(f'<div class="section-title" style="margin-top: 1.5rem;">{t("dataset_preview")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="chart-caption" style="text-align: left; margin-top: -0.7rem;">{t("dataset_caption")}</div>',
        unsafe_allow_html=True,
    )

    display_frame = dataset.copy()
    display_frame["Date"] = pd.to_datetime(display_frame["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_frame, use_container_width=True, hide_index=True, height=560)


def render_variable_page() -> None:
    render_header(
        title=t("variable_title"),
        subtitle=t("variable_subtitle"),
    )

    with st.spinner(t("variable_loading")):
        dataset = get_dataset_frame()

    variable_catalog = build_variable_catalog(dataset)

    st.html(
        """
        <div class="variable-intro">
            <strong>{goal}</strong> {intro}
        </div>
        """.format(goal=t("variable_goal"), intro=t("variable_intro"))
    )

    if current_language() == "en":
        variable_cards = [
            {
                "icon": "₿",
                "title": "Crypto Prices (Bitcoin, Ethereum, Solana)",
                "description": "Historical closing prices of the coins used in this research: Bitcoin, Ethereum, and Solana.",
                "source": "Binance API",
                "relevance": (
                    "The closing prices of Bitcoin, Ethereum, and Solana are the main variables used as prediction targets. "
                    "Historical prices are processed into daily returns to capture price movement more clearly and reduce scale differences. "
                    "Moving averages with 7, 14, and 30-day windows are also used to represent short to medium-term trends, while lagged price "
                    "and return variables help the models understand how previous price behavior may influence current movement."
                ),
            },
            {
                "icon": "▥",
                "title": "Market Volume",
                "description": (
                    "Market volume represents total crypto market trading activity, built from the aggregated daily trading volume of "
                    "Bitcoin, Ethereum, and Solana."
                ),
                "source": "Binance API",
                "relevance": (
                    "Market volume is used as an external indicator to capture liquidity and overall crypto market activity. Volume changes often "
                    "relate to price movement, either as a signal of trend strength or rising volatility. Including market volume helps SARIMAX "
                    "and XGBoost understand broader market conditions."
                ),
            },
            {
                "icon": "▮",
                "title": "Market Volatility",
                "description": (
                    "Volatility measures the fluctuation level of crypto asset prices, calculated from daily returns using rolling standard deviation."
                ),
                "source": "Binance API",
                "relevance": (
                    "Volatility represents uncertainty and risk in Bitcoin, Ethereum, and Solana price movements. In this research, volatility is "
                    "calculated using 7, 14, and 30-day rolling windows to capture short to medium-term risk dynamics. This information helps the "
                    "models adjust predictions under changing market conditions."
                ),
            },
            {
                "icon": "△",
                "title": "Fear and Greed Index",
                "description": (
                    "The Fear and Greed Index is a market sentiment indicator that reflects crypto market psychology, ranging from fear to euphoria."
                ),
                "source": "Fear and Greed Index API",
                "relevance": (
                    "The Fear and Greed Index is normalized to a 0-1 range for XGBoost so its scale is comparable with other numerical features. "
                    "For SARIMAX, it is used without normalization because the model is less sensitive to scale differences and can preserve the "
                    "original sentiment interpretation."
                ),
            },
        ]
    else:
        variable_cards = [
        {
            "icon": "₿",
            "title": "Harga Kripto (Bitcoin, Ethereum, Solana)",
            "description": "Histori harga closing dari koin yang digunakan untuk penelitian ini, yaitu Bitcoin, Ethereum, dan Solana.",
            "source": "Binance API",
            "relevance": (
                "Harga penutupan Bitcoin, Ethereum, dan Solana merupakan variabel utama yang menjadi target prediksi dalam penelitian ini. "
                "Data harga historis diolah melalui perhitungan return harian untuk menangkap perubahan harga secara lebih representatif, "
                "serta mengurangi perbedaan skala data. Selain itu, digunakan fitur moving average dengan window 7, 14, dan 30 hari untuk "
                "merepresentasikan tren jangka pendek hingga menengah, serta lag features harga dan return untuk menangkap pengaruh nilai "
                "historis terhadap pergerakan harga saat ini."
            ),
        },
        {
            "icon": "▥",
            "title": "Volume Market",
            "description": (
                "Market volume merupakan variabel yang merepresentasikan total aktivitas perdagangan pasar kripto, "
                "yang dibentuk dari agregasi volume transaksi harian Bitcoin, Ethereum, dan Solana."
            ),
            "source": "Binance API",
            "relevance": (
                "Market volume digunakan sebagai indikator eksternal untuk menangkap tingkat likuiditas dan intensitas aktivitas pasar kripto "
                "secara keseluruhan. Perubahan volume sering kali berkaitan dengan perubahan harga, baik sebagai sinyal penguatan tren maupun "
                "peningkatan volatilitas. Dengan memasukkan market volume ke dalam model, SARIMAX dan XGBoost diharapkan mampu memahami kondisi "
                "pasar secara lebih komprehensif."
            ),
        },
        {
            "icon": "▮",
            "title": "Volatilitas Market",
            "description": (
                "Volatilitas merupakan ukuran tingkat fluktuasi harga aset kripto yang dihitung berdasarkan return harian "
                "menggunakan rolling standard deviation."
            ),
            "source": "Binance API",
            "relevance": (
                "Volatilitas digunakan untuk merepresentasikan tingkat ketidakpastian dan risiko pergerakan harga Bitcoin, Ethereum, dan Solana "
                "yang dikenal sangat fluktuatif. Dalam penelitian ini, volatilitas dihitung menggunakan rolling window 7, 14, dan 30 hari untuk "
                "menangkap dinamika risiko dalam jangka pendek hingga menengah. Informasi volatilitas ini penting karena perubahan tingkat "
                "ketidakpastian pasar sering memengaruhi pola pergerakan harga."
            ),
        },
        {
            "icon": "△",
            "title": "Indeks Fear and Greed",
            "description": (
                "Indeks Fear and Greed merupakan indikator sentimen pasar yang mencerminkan kondisi psikologis pelaku pasar kripto, "
                "mulai dari rasa takut hingga euforia."
            ),
            "source": "Fear and Greed Index API",
            "relevance": (
                "Dalam penelitian ini, Fear and Greed Index dinormalisasi ke rentang 0-1 untuk digunakan pada model XGBoost agar memiliki skala "
                "yang sebanding dengan fitur numerik lainnya. Sementara itu, pada model SARIMAX, Fear and Greed Index digunakan tanpa normalisasi "
                "karena model ini tidak sensitif terhadap perbedaan skala variabel dan mempertahankan interpretasi nilai sentimen aslinya."
            ),
        },
    ]
    cards_markup = "\n".join(
        f"""
        <section class="variable-info-card">
            <div class="variable-card-title">
                <span class="variable-card-icon">{item["icon"]}</span>
                <span>{item["title"]}</span>
            </div>
            <div class="variable-card-divider"></div>
            <div class="variable-section-label">{t("description")}</div>
            <p>{item["description"]}</p>
            <div class="variable-section-label">{t("data_source_label")}</div>
            <p>{item["source"]}</p>
            <div class="variable-section-label">{t("prediction_relevance")}</div>
            <p>{item["relevance"]}</p>
        </section>
        """
        for item in variable_cards
    )
    st.html(f'<div class="variable-card-stack">{cards_markup}</div>')

    st.markdown(f'<div class="section-title" style="margin-top: 1.5rem;">{t("final_variables")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="chart-caption" style="text-align: left; margin-top: -0.7rem;">{t("final_variables_caption")}</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(variable_catalog, use_container_width=True, hide_index=True, height=520)


def render_method_page() -> None:
    render_header(title=t("method_title"), subtitle=t("method_subtitle"))

    if current_language() == "en":
        labels = {
            "intro": "Why Two Models?",
            "definition": "Definition",
            "usage": "What It Is Used For",
            "formula": "Formula",
            "symbols": "Meaning of Symbols",
            "reason": "Why This Model Was Selected",
            "sarimax_formula": "yₜ = c + φyₜ₋₁ + θeₜ₋₁ + βXₜ + εₜ",
            "xgboost_formula": "ŷᵢ = Σₖ₌₁ᴷ fₖ(xᵢ),   Obj = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)",
        }
        sarimax = {
            "definition": "SARIMAX stands for Seasonal AutoRegressive Integrated Moving Average with eXogenous variables. In simple terms, it is a time-series model that studies how today’s price is related to previous prices, previous prediction errors, seasonal patterns, and outside factors.",
            "usage": "It is used to forecast future prices when the order of time matters. For crypto, this means the model looks at historical price movement and external indicators to estimate the next price values.",
            "reason": "SARIMAX was selected because crypto prices move as a sequence over time. The model is useful for capturing trend, short-term momentum, repeated patterns, and the influence of external variables while still being relatively interpretable.",
            "symbols": [
                ("yₜ", "crypto price at time t"),
                ("c", "constant or base value"),
                ("φ", "effect of previous price values"),
                ("θ", "effect of previous forecast errors"),
                ("βXₜ", "effect of external variables such as volume, volatility, and Fear & Greed"),
                ("εₜ", "unexplained random error"),
            ],
        }
        xgboost = {
            "definition": "XGBoost is a machine learning model based on many decision trees. Each tree tries to correct mistakes made by earlier trees, so the final prediction is built step by step.",
            "usage": "It is used to learn complex relationships between many variables, such as price history, moving averages, volume, volatility, and sentiment indicators.",
            "reason": "XGBoost was selected because crypto market behavior can be non-linear. A small change in volume or volatility may have a different impact depending on the market condition, and XGBoost is strong at learning that kind of pattern.",
            "symbols": [
                ("ŷᵢ", "predicted price for observation i"),
                ("K", "number of decision trees"),
                ("fₖ(xᵢ)", "prediction contribution from tree k using input variables xᵢ"),
                ("Obj", "training objective that balances prediction error and model complexity"),
                ("l", "loss or error between actual and predicted price"),
                ("Ω", "penalty that keeps the model from becoming too complicated"),
            ],
        }
    else:
        labels = {
            "intro": "Mengapa Menggunakan Dua Model?",
            "definition": "Definisi",
            "usage": "Digunakan Untuk Apa",
            "formula": "Rumus",
            "symbols": "Arti Simbol",
            "reason": "Alasan Model Dipilih",
            "sarimax_formula": "yₜ = c + φyₜ₋₁ + θeₜ₋₁ + βXₜ + εₜ",
            "xgboost_formula": "ŷᵢ = Σₖ₌₁ᴷ fₖ(xᵢ),   Obj = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)",
        }
        sarimax = {
            "definition": "SARIMAX adalah singkatan dari Seasonal AutoRegressive Integrated Moving Average with eXogenous variables. Secara sederhana, SARIMAX adalah model deret waktu yang membaca hubungan antara harga hari ini dengan harga sebelumnya, kesalahan prediksi sebelumnya, pola musiman, dan faktor luar.",
            "usage": "Model ini digunakan untuk memprediksi harga ketika urutan waktu sangat penting. Pada kripto, SARIMAX melihat pergerakan harga historis dan indikator eksternal untuk memperkirakan nilai harga berikutnya.",
            "reason": "SARIMAX dipilih karena harga kripto bergerak secara berurutan dari waktu ke waktu. Model ini membantu menangkap tren, momentum jangka pendek, pola berulang, serta pengaruh variabel eksternal dengan hasil yang masih relatif mudah dijelaskan.",
            "symbols": [
                ("yₜ", "harga kripto pada waktu ke-t"),
                ("c", "nilai dasar atau konstanta"),
                ("φ", "pengaruh harga pada periode sebelumnya"),
                ("θ", "pengaruh kesalahan prediksi pada periode sebelumnya"),
                ("βXₜ", "pengaruh variabel eksternal seperti volume, volatilitas, dan Fear & Greed"),
                ("εₜ", "error acak yang tidak dapat dijelaskan model"),
            ],
        }
        xgboost = {
            "definition": "XGBoost adalah model machine learning yang menggunakan banyak pohon keputusan. Setiap pohon berusaha memperbaiki kesalahan dari pohon sebelumnya, sehingga prediksi akhir dibangun secara bertahap.",
            "usage": "Model ini digunakan untuk mempelajari hubungan yang kompleks antara banyak variabel, seperti harga historis, moving average, volume, volatilitas, dan indikator sentimen.",
            "reason": "XGBoost dipilih karena perilaku pasar kripto sering tidak linear. Perubahan kecil pada volume atau volatilitas bisa berdampak berbeda tergantung kondisi pasar, dan XGBoost kuat dalam mempelajari pola seperti itu.",
            "symbols": [
                ("ŷᵢ", "harga prediksi untuk data ke-i"),
                ("K", "jumlah pohon keputusan"),
                ("fₖ(xᵢ)", "kontribusi prediksi dari pohon ke-k berdasarkan variabel input xᵢ"),
                ("Obj", "tujuan pelatihan yang menyeimbangkan error prediksi dan kompleksitas model"),
                ("l", "loss atau selisih antara harga aktual dan harga prediksi"),
                ("Ω", "penalti agar model tidak terlalu rumit"),
            ],
        }

    def symbol_rows(items: list[tuple[str, str]]) -> str:
        return "".join(
            f"<div class='method-symbol-row'><span>{html.escape(symbol)}</span><p>{html.escape(description)}</p></div>"
            for symbol, description in items
        )

    def model_card(title: str, badge: str, formula: str, content: dict[str, object]) -> str:
        return f"""
        <section class="method-card">
            <div class="method-card-header">
                <p class="method-badge">{html.escape(badge)}</p>
                <h3>{html.escape(title)}</h3>
            </div>
            <div class="method-grid">
                <div class="method-panel">
                    <h4>{html.escape(labels["definition"])}</h4>
                    <p>{html.escape(str(content["definition"]))}</p>
                </div>
                <div class="method-panel">
                    <h4>{html.escape(labels["usage"])}</h4>
                    <p>{html.escape(str(content["usage"]))}</p>
                </div>
            </div>
            <div class="method-formula">
                <h4>{html.escape(labels["formula"])}</h4>
                <code>{html.escape(formula)}</code>
            </div>
            <div class="method-symbols">
                <h4>{html.escape(labels["symbols"])}</h4>
                {symbol_rows(content["symbols"])}
            </div>
            <div class="method-panel method-reason">
                <h4>{html.escape(labels["reason"])}</h4>
                <p>{html.escape(str(content["reason"]))}</p>
            </div>
        </section>
        """

    st.html(
        f"""
        <section class="method-intro">
            <div class="method-intro-icon">📚</div>
            <div>
                <p class="method-badge">{html.escape(labels["intro"])}</p>
                <p>{html.escape(t("method_intro_body"))}</p>
            </div>
        </section>
        <div class="method-page">
            {model_card("SARIMAX", "Time Series Model", labels["sarimax_formula"], sarimax)}
            {model_card("XGBoost", "Machine Learning Model", labels["xgboost_formula"], xgboost)}
        </div>
        """
    )


def render_placeholder_page(title: str) -> None:
    render_header(title=title, subtitle=t("placeholder_subtitle"))
    st.markdown(
        f'<div class="placeholder">{t("placeholder_body").format(title=title)}</div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <footer class="app-footer">
            <div class="footer-left">
                <span>© 2025 Thaddeus Kendrick Andrian</span>
                <span class="footer-separator">•</span>
                <span>2602207054</span>
                <span class="footer-separator">•</span>
                <span>Computer Science and Statistics</span>
                <span class="footer-separator">•</span>
                <span>Universitas Bina Nusantara</span>
            </div>
            <div class="footer-right">
                <span>Connect with me:</span>
                <a class="footer-linkedin" href="https://www.linkedin.com/in/thaddeuska/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn Thaddeus Kendrick Andrian">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.34V9h3.41v1.56h.05c.47-.9 1.64-1.85 3.37-1.85 3.61 0 4.28 2.38 4.28 5.47v6.27ZM5.32 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12Zm1.78 13.02H3.54V9H7.1v11.45ZM22.22 0H1.77C.79 0 0 .77 0 1.72v20.56C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.72V1.72C24 .77 23.2 0 22.22 0Z"/>
                    </svg>
                    <span>LinkedIn</span>
                </a>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.html(load_css())
    render_page_loader()
    render_mobile_nav_toggle()
    selected_page = build_sidebar()
    render_footer()

    if selected_page == "prediksi":
        render_prediction_page()
    elif selected_page == "komparasi":
        render_comparison_page()
    elif selected_page == "variabel":
        render_variable_page()
    elif selected_page == "metode":
        render_method_page()
    elif selected_page == "backtesting":
        render_backtesting_page()
    else:
        render_dataset_page()


if __name__ == "__main__":
    main()
