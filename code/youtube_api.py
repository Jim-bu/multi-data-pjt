import googleapiclient.discovery
import pandas as pd
import isodate
import streamlit as st
from tqdm import tqdm
from stqdm import stqdm
from googleapiclient.errors import HttpError

# YouTube API 서비스를 생성하는 함수
def get_youtube_service(api_key):
    return googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

# 특정 비디오의 정보를 가져오는 함수
def get_video_info(youtube, video_id):
    try:
        request = youtube.videos().list(
            part='snippet,contentDetails',
            id=video_id,
            fields='items(id,snippet(title,channelId,channelTitle,publishedAt),contentDetails(duration))'
        )
        response = request.execute()
        return response['items'][0]
    except Exception as e:
        st.write(f"Error fetching video info for {video_id}: {e}")
        return None

# 특정 채널의 정보를 가져오는 함수
def get_channel_info(youtube, channel_id):
    try:
        request = youtube.channels().list(
            part='snippet,statistics',
            id=channel_id,
            fields='items(statistics(subscriberCount))'
        )
        response = request.execute()
        return response['items'][0]['statistics']
    except Exception as e:
        st.write(f"Error fetching channel info for {channel_id}: {e}")
        return None

# 채널 이름을 기반으로 채널 ID를 가져오는 함수
def get_channel_id_by_name(youtube, channel_name):
    try:
        request = youtube.search().list(
            part='snippet',
            q=channel_name,
            type='channel',
            fields='items(id(channelId),snippet(channelTitle))'
        )
        response = request.execute()
        return response['items'][0]['id']['channelId']
    except Exception as e:
        st.write(f"Error fetching channel ID for {channel_name}: {e}")
        return None

# 특정 채널의 비디오 ID를 가져오는 함수
def get_video_ids_from_channel(youtube, channel_id):
    video_ids = []
    page_token = None
    while True:
        try:
            request = youtube.search().list(
                part='id',
                channelId=channel_id,
                maxResults=50,
                pageToken=page_token,
                type='video'
            )
            response = request.execute()
            video_ids += [item['id']['videoId'] for item in response['items']]
            if 'nextPageToken' in response:
                page_token = response['nextPageToken']
            else:
                break
        except HttpError as e:
            st.write(f"Error fetching video IDs for channel {channel_id}: {e}")
            break
        except Exception as e:
            st.write(f"Error fetching video IDs for channel {channel_id}: {e}")
            break
    return video_ids

# 동영상의 댓글 정보를 가져오는 함수 (페이지네이션 적용)
def get_all_comments(youtube, video_id):
    comments = []
    page_token = None

    while True:
        try:
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
                pageToken=page_token
            )
            response = request.execute()

            for item in response['items']:
                try:
                    top_comment = item['snippet']['topLevelComment']['snippet']
                    comment = {
                        'comment_thread_id': item['id'],
                        'video_id': video_id,
                        'author_display_name': top_comment['authorDisplayName'],
                        'published_at': format_datetime(top_comment['publishedAt']),
                        'author_channel_id': top_comment['authorChannelId']['value'] if 'authorChannelId' in top_comment else None
                    }
                    comments.append(comment)

                    if 'replies' in item:
                        reply_count = item['snippet']['totalReplyCount']
                        if reply_count > 0:
                            reply_page_token = None
                            while True:
                                try:
                                    reply_request = youtube.comments().list(
                                        part='snippet',
                                        parentId=item['id'],
                                        maxResults=min(reply_count, 100),
                                        pageToken=reply_page_token
                                    )
                                    reply_response = reply_request.execute()

                                    for reply_item in reply_response['items']:
                                        try:
                                            reply_comment = {
                                                'comment_thread_id': reply_item['id'],
                                                'video_id': video_id,
                                                'author_display_name': reply_item['snippet']['authorDisplayName'],
                                                'published_at': format_datetime(reply_item['snippet']['publishedAt']),
                                                'author_channel_id': reply_item['snippet']['authorChannelId']['value'] if 'authorChannelId' in reply_item['snippet'] else None
                                            }
                                            comments.append(reply_comment)
                                        except KeyError as e:
                                            st.write(f"KeyError in processing reply: {e}")

                                    if 'nextPageToken' in reply_response:
                                        reply_page_token = reply_response['nextPageToken']
                                    else:
                                        break

                                except HttpError as e:
                                    st.write(f"Error fetching replies for comment {item['id']}: {e}")
                                    break

                                except Exception as e:
                                    st.write(f"Error fetching replies for comment {item['id']}: {e}")
                                    break

                except KeyError as e:
                    st.write(f"KeyError in processing comment: {e}")

            if 'nextPageToken' in response:
                page_token = response['nextPageToken']
            else:
                break

        except HttpError as e:
            st.write(f"Error fetching comments for video {video_id}: {e}")
            break

        except Exception as e:
            st.write(f"Error fetching comments for video {video_id}: {e}")
            break

    return comments

# 동영상 길이를 확인하여 short 비디오인지 여부를 판단하는 함수
def is_short(video_duration):
    try:
        duration = isodate.parse_duration(video_duration)
        return duration.total_seconds() <= 60
    except Exception as e:
        st.write(f"Error parsing duration: {e}")
        return False

# 날짜 및 시간을 지정된 형식으로 변환하는 함수
def format_datetime(datetime_str):
    try:
        return pd.to_datetime(datetime_str).strftime('%Y-%m-%dT%H:%M')
    except Exception as e:
        st.write(f"Error formatting datetime: {e}")
        return datetime_str

# 데이터 수집 함수
def collect_data(youtube, video_ids, start_index, end_index):
    combined_data = []
    selected_video_ids = video_ids[start_index:end_index]

    for video_id in stqdm(selected_video_ids, desc="Processing videos"):
        video_info = get_video_info(youtube, video_id)
        if video_info:
            channel_id = video_info['snippet']['channelId']
            channel_author = video_info['snippet']['channelTitle']
            title = video_info['snippet']['title']
            duration = video_info['contentDetails']['duration']
            upload_date = format_datetime(video_info['snippet']['publishedAt'])

            # 동영상 기본 정보 추가
            combined_data.append({
                'Channel_ID': channel_id,
                'Channel_Title': channel_author,
                'Video_ID': video_id,
                'Title': title,
                'Upload_Date': upload_date,
                'Comment_ID': None,
                'Comment_Author': None,
                'Is_Channel_Author': True,
                'Comment_Published_At': None,
                'Is_Short': is_short(duration)
            })

            # 댓글 데이터를 추가
            comments = get_all_comments(youtube, video_id)
            for comment in comments:
                combined_data.append({
                    'Channel_ID': channel_id,
                    'Channel_Title': channel_author,
                    'Video_ID': video_id,
                    'Title': title,
                    'Upload_Date': upload_date,
                    'Comment_ID': comment['comment_thread_id'],
                    'Comment_Author': comment['author_display_name'],
                    'Is_Channel_Author': (comment['author_channel_id'] == channel_id),
                    'Comment_Published_At': comment['published_at'],
                    'Is_Short': is_short(duration)
                })

    return combined_data

# 데이터 저장 함수
def save_data(data, file_name):
    try:
        df_combined = pd.DataFrame(data)
        df_combined.to_csv(f'{file_name}.csv', index=False)
        st.write("데이터를 저장했습니다.")
    except Exception as e:
        st.write(f"Error saving data: {e}")

# 메인 함수
def main():
    st.title("YouTube Data Collector")

    api_key = st.text_input("API 키를 입력하세요", type="password")
    if not api_key:
        st.stop()

    youtube = get_youtube_service(api_key)

    channel_name = st.text_input("채널명을 입력하세요")
    if not channel_name:
        st.stop()

    channel_id = get_channel_id_by_name(youtube, channel_name)
    if not channel_id:
        st.write(f"채널 '{channel_name}'을 찾을 수 없습니다.")
        return

    video_ids = get_video_ids_from_channel(youtube, channel_id)
    st.write(f"채널 '{channel_name}'에서 {len(video_ids)}개의 비디오를 찾았습니다.")

    start_index = st.number_input("비디오 ID를 가져올 시작 인덱스를 입력하세요 (0부터 시작)", min_value=0, max_value=len(video_ids)-1, value=0)
    end_index = st.number_input("비디오 ID를 가져올 끝 인덱스를 입력하세요 (마지막 비디오 ID는 포함되지 않음)", min_value=start_index+1, max_value=len(video_ids), value=len(video_ids))

    file_name = st.text_input("저장할 파일명을 입력하세요 (확장자 제외)")
    if not file_name:
        st.stop()

    if st.button("데이터 수집 시작"):
        data = collect_data(youtube, video_ids, start_index, end_index)
        save_data(data, file_name)

if __name__ == "__main__":
    main()
