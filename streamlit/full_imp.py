import streamlit as st
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 데이터 처리 함수
def processing1(data):
    data['Upload_Date'] = pd.to_datetime(data['Upload_Date'])
    data['Comment_Published_At'] = pd.to_datetime(data['Comment_Published_At'])
    data = data.dropna(subset=['Comment_ID'])
    data = data.drop_duplicates(subset=['Comment_ID'])
    data = data[data['Is_Channel_Author'] != 1]
    data = data.drop_duplicates(subset=['Video_ID','Comment_Author'])
    data = data.reset_index(drop=True)
    remaining_comment_count = data['Comment_ID'].nunique()
    st.write(f"채널 주인의 댓글을 제외하고, 영상별로 작성자의 댓글을 1개씩만 남긴 후의 댓글 ID 개수는 {remaining_comment_count}개 입니다.")

    columns_to_use = [
        'Channel_ID', 'Channel_Title', 'Upload_Date', 'Comment_Published_At', 
        'Video_ID', 'Comment_ID'
    ]
    data = data[columns_to_use].copy()

    data['Upload_Date'] = data['Upload_Date'].astype(str).str[:10]
    data['Comment_Published_At'] = data['Comment_Published_At'].astype(str).str[:10]
    data['Upload_Date'] = pd.to_datetime(data['Upload_Date'], format='%Y-%m-%d', errors='coerce')
    data['Comment_Published_At'] = pd.to_datetime(data['Comment_Published_At'], format='%Y-%m-%d', errors='coerce')
    
    df = pd.DataFrame()
    
    try:
        for channel_id, channel_data in data.groupby('Channel_ID'):
            first_upload_date = channel_data['Upload_Date'].min()
            date_range = pd.date_range(start=first_upload_date, end=channel_data['Upload_Date'].max() + pd.Timedelta(days=7), freq='W-MON')
            weekly_data = []
            cumulative_videos = 0
            cumulative_comments = 0

            for i, week_start in enumerate(date_range):
                week_end = week_start + pd.Timedelta(days=7)
                weekly_videos = channel_data[(channel_data['Upload_Date'] >= week_start) & (channel_data['Upload_Date'] < week_end)]
                weekly_comments = channel_data[(channel_data['Comment_Published_At'] >= week_start) & (channel_data['Comment_Published_At'] < week_end)]
                unique_comments = weekly_comments.drop_duplicates(subset=['Comment_ID', 'Comment_Published_At']).shape[0]
                cumulative_videos += weekly_videos['Video_ID'].nunique()
                cumulative_comments += unique_comments
                weekly_data.append({
                    'Channel_ID': channel_id,
                    'Channel_Title': channel_data['Channel_Title'].iloc[0],
                    'Comment_Year': week_start.year,
                    'Comment_Week': week_start.isocalendar()[1],
                    'Cumulative_Videos': int(cumulative_videos),
                    'Cumulative_Comments': int(cumulative_comments),
                })
            channel_df = pd.DataFrame(weekly_data)
            df = pd.concat([df, channel_df], ignore_index=True)
    except Exception as e:
        st.write(f"데이터 처리 중 오류 발생: {e}")
    
    if 'Cumulative_Videos' in df.columns:
        df = df[(df['Cumulative_Videos'] != 0)]
    else:
        st.write("'Cumulative_Videos' 열이 결과 데이터프레임에 존재하지 않습니다.")
    
    expected_columns = ['Channel_ID', 'Channel_Title', 'Comment_Year', 'Comment_Week', 'Cumulative_Videos', 'Cumulative_Comments']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        st.write(f"다음 열이 결과 데이터프레임에 없습니다: {missing_columns}")
    else:
        df = df[expected_columns]
    df['Comments_Per_Video'] = df['Cumulative_Comments'] / df['Cumulative_Videos']
    df['Comments_Per_Video'] = df['Comments_Per_Video'].round(1)  # 소숫점 첫째 자리에서 반올림
    df['Year_Week'] = df['Comment_Year'].astype(str) + '-' + df['Comment_Week'].astype(str)
    df = df.drop(['Channel_ID','Comment_Year','Comment_Week'],axis=1)
    
    return df

# 슬라이딩 윈도우와 기울기 계산 함수
def calculate_slopes(data, periods):
    if not isinstance(periods, int):
        raise ValueError("`periods` must be an integer.")

    slopes = []
    channels = data['Channel_Title'].unique()

    for period in tqdm([periods], desc='Creating sliding windows and calculating slopes'):
        for channel in channels:
            channel_data = data[data['Channel_Title'] == channel]
            if len(channel_data) >= period:
                for i in range(len(channel_data) - period + 1):
                    window = channel_data.iloc[i:i+period]
                    comments_per_video = window['Comments_Per_Video'].values
                    year_week = window['Year_Week'].values[-1]

                    x = np.arange(len(comments_per_video)).reshape(-1, 1)
                    y = comments_per_video

                    model = LinearRegression().fit(x, y)
                    slope = model.coef_[0]

                    slopes.append({
                        'Channel_Title': channel,
                        'Period': period,
                        'Year_Week': year_week,
                        'Window': comments_per_video.round(1),  # 소숫점 첫째 자리에서 반올림
                        'Slope': round(slope, 1)  # 소숫점 첫째 자리에서 반올림
                    })
        print(f"Period {period}: {len(slopes)} slopes calculated")

    return pd.DataFrame(slopes)

# 기울기 계산 후 결과 처리 함수
def process_data(df1, df2, threshold1, threshold2):
    channels = set(df1['Channel_Title'].unique()).union(set(df2['Channel_Title'].unique()))

    for channel in channels:
        st.write(f'<b>{channel}</b>', unsafe_allow_html=True)
        
        channel_data_long = df2[df2['Channel_Title'] == channel]
        if not channel_data_long.empty:
            max_slope_row = channel_data_long[channel_data_long['Slope'] == channel_data_long['Slope'].max()]
            max_slope = max_slope_row['Slope'].values[0]
            max_year_week = max_slope_row['Year_Week'].values[0]
            
            st.write(f'롱폼의 경우')
            st.write(f'가장 큰 기울기: <span style="color:blue;">{max_slope}</span> (기간: {max_year_week})', unsafe_allow_html=True)
            if max_slope > threshold2:
                st.write(f'<span style="color:red; font-weight:bold;">{channel}는 떡상 기준에 적합합니다!</span> (기울기: {max_slope} > {threshold2})', unsafe_allow_html=True)
            else:
                st.write(f'<span style="color:gray;">{channel}는 떡상 기준에 부적합합니다!</span> (기울기: {max_slope} < {threshold2})', unsafe_allow_html=True)
        
        channel_data_short = df1[df1['Channel_Title'] == channel]
        if not channel_data_short.empty:
            max_slope_row = channel_data_short[channel_data_short['Slope'] == channel_data_short['Slope'].max()]
            max_slope = max_slope_row['Slope'].values[0]
            max_year_week = max_slope_row['Year_Week'].values[0]
            
            st.write(f'쇼츠의 경우')
            st.write(f'가장 큰 기울기: <span style="color:blue;">{max_slope}</span> (기간: {max_year_week})', unsafe_allow_html=True)
            if max_slope > threshold1:
                st.write(f'<span style="color:red; font-weight:bold;">{channel}는 떡상 기준에 적합합니다!</span> (기울기: {max_slope} > {threshold1})', unsafe_allow_html=True)
            else:
                st.write(f'<span style="color:gray;">{channel}는 떡상 기준에 부적합합니다!</span> (기울기: {max_slope} < {threshold1})', unsafe_allow_html=True)
        
        st.write('<hr>', unsafe_allow_html=True)

# 채널 분석 함수
def analyze_channel1(channel_name, df1, df2):
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우의 경우
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    st.write(f'<b>{channel_name}</b>', unsafe_allow_html=True)

    if not df2[df2.Channel_Title == channel_name].empty:
        st.write('롱폼의 경우')
        df_long = df2[df2.Channel_Title == channel_name].copy()
        df_long = df_long.reset_index(drop=True)
        
        signal = df_long['Comments_Per_Video']
        
        df_long['4_Week_Moving_Avg'] = signal.rolling(window=4).mean().round(1)  # 소숫점 첫째 자리에서 반올림
        df_long['Weekly_Moving_Avg'] = signal.rolling(window=1).mean().round(1)  # 소숫점 첫째 자리에서 반올림
        df_long['3_Week_Moving_Avg'] = signal.rolling(window=3).mean().round(1)  # 소숫점 첫째 자리에서 반올림

        df_long['Pct_Change'] = signal.pct_change().round(1)  # 소숫점 첫째 자리에서 반올림
        
        df_long_filtered = df_long.copy()
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(df_long_filtered['Year_Week'], df_long_filtered['Comments_Per_Video'], label='Comments Per Video', marker='o')
        plt.plot(df_long_filtered['Year_Week'], df_long_filtered['4_Week_Moving_Avg'], label='4-Week Moving Average', linestyle='--')
        plt.plot(df_long_filtered['Year_Week'], df_long_filtered['Weekly_Moving_Avg'], label='Weekly Moving Average', linestyle='-.')
        plt.plot(df_long_filtered['Year_Week'], df_long_filtered['3_Week_Moving_Avg'], label='3-Week Moving Average', linestyle=':')
        
        plt.xlabel('Year-Week')
        plt.ylabel('Comments Per Video')
        plt.title(f'{channel_name}: Comments Per Video Analysis (Long Form)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid()
        
        st.pyplot(plt)
        
    if not df1[df1.Channel_Title == channel_name].empty:
        st.write('쇼츠의 경우')
        df_short = df1[df1.Channel_Title == channel_name].copy()
        df_short = df_short.reset_index(drop=True)
        
        signal = df_short['Comments_Per_Video']
        
        df_short['4_Week_Moving_Avg'] = signal.rolling(window=4).mean().round(1)  # 소숫점 첫째 자리에서 반올림
        df_short['Weekly_Moving_Avg'] = signal.rolling(window=1).mean().round(1)  # 소숫점 첫째 자리에서 반올림
        df_short['3_Week_Moving_Avg'] = signal.rolling(window=3).mean().round(1)  # 소숫점 첫째 자리에서 반올림

        df_short['Pct_Change'] = signal.pct_change().round(1)  # 소숫점 첫째 자리에서 반올림
        
        df_short_filtered = df_short.copy()
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(df_short_filtered['Year_Week'], df_short_filtered['Comments_Per_Video'], label='Comments Per Video', marker='o')
        plt.plot(df_short_filtered['Year_Week'], df_short_filtered['4_Week_Moving_Avg'], label='4-Week Moving Average', linestyle='--')
        plt.plot(df_short_filtered['Year_Week'], df_short_filtered['Weekly_Moving_Avg'], label='Weekly Moving Average', linestyle='-.')
        plt.plot(df_short_filtered['Year_Week'], df_short_filtered['3_Week_Moving_Avg'], label='3-Week Moving Average', linestyle=':')
        
        plt.xlabel('Year-Week')
        plt.ylabel('Comments Per Video')
        plt.title(f'{channel_name}: Comments Per Video Analysis (Shorts)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid()
        
        st.pyplot(plt)

# 스트림릿 앱 시작
st.title('YouTube Comment Data Processing')

# 페이지 관리
if 'page' not in st.session_state:
    st.session_state.page = 1

# 페이지 내용 렌더링
if st.session_state.page == 1:
    st.header('CSV 파일 업로드')
    uploaded_file = st.file_uploader('CSV 파일을 업로드하세요', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        data = df.copy()
            
        columns_to_use = ['Channel_ID', 'Channel_Title','Video_ID', 'Title', 'Upload_Date','Is_Short',
            'Comment_ID', 'Comment_Published_At', 'Comment_Author', 'Is_Channel_Author']
    
        data = data[columns_to_use].copy()
        data['Upload_Date'] = pd.to_datetime(data['Upload_Date'])
        data['Comment_Published_At'] = pd.to_datetime(data['Comment_Published_At'])

        data1 = data[data['Is_Short'] == True].copy()
        data2 = data[data['Is_Short'] == False].copy()

        result_data1 = processing1(data1)
        result_data2 = processing1(data2)

        st.session_state.result_data1 = result_data1
        st.session_state.result_data2 = result_data2

        st.subheader('Processed Data for Shorts')
        st.dataframe(result_data1)

        st.subheader('Processed Data for Non-Shorts')
        st.dataframe(result_data2)

        if st.button('다음 페이지로 이동'):
            st.session_state.page = 2

elif st.session_state.page == 2:
    st.header('기울기 계산 결과')
    period = 2
    data1_slopes = calculate_slopes(st.session_state.result_data1, period)
    data2_slopes = calculate_slopes(st.session_state.result_data2, period)

    process_data(data1_slopes, data2_slopes, 158.5, 238.18)

    if st.button('다음 페이지로 이동'):
        st.session_state.page = 3

elif st.session_state.page == 3:
    st.header('채널별 댓글 분석')
    dc1 = st.session_state.result_data1['Channel_Title'].unique()
    dc2 = st.session_state.result_data2['Channel_Title'].unique()

    for channel in dc1:
        analyze_channel1(channel, st.session_state.result_data1, st.session_state.result_data2)
    
    for channel in dc2:
        if channel not in dc1:
            analyze_channel1(channel, st.session_state.result_data1, st.session_state.result_data2)

    if st.button('처음 페이지로 이동'):
        st.session_state.page = 1
