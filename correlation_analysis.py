import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
import datetime
# import pickle
import streamlit as st
import model_building as m
import technical_analysis as t
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def corr_plot(combined_df):

    pct_chg_df = combined_df.pct_change() * 100
    pct_chg_df.dropna(inplace=True, how='any', axis=0)
    st.title("Correlation Analysis of Stocks")
    pct_chg_df = pct_chg_df.to_frame()   
    numeric_cols = pct_chg_df.select_dtypes(include=[np.number]).columns
    corr_matrix = pct_chg_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    st.pyplot(fig)
