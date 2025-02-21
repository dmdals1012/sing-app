import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import librosa.display
import io
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue
import time

# 모델 로드
@st.cache_resource
def load_model():
    return keras.models.load_model('vocal_range_classifier.h5')

model = load_model()

# 레이블 인코더 로드
label_encoder = joblib.load('label_encoder.pkl')

st.title('실시간 음역대 분류기')

# 세션 상태 초기화
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = 'stopped'

audio_buffer = queue.Queue()

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)  # 평균값으로 고정된 크기의 벡터 생성
    return mfcc

# 오디오 프레임 처리

def audio_receiver(frame: av.AudioFrame):
    if st.session_state.recording_state == 'recording':
        sound = np.array(frame.to_ndarray()).flatten().astype(np.float32) / 32768.0
        audio_buffer.put(sound)

webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:numb.viagenie.ca"],
                "username": "webrtc@live.com",
                "credential": "muazkh"
            }
        ]
    },
    media_stream_constraints={"video": False, "audio": True},
    on_audio_frame=audio_receiver,
)

if webrtc_ctx and webrtc_ctx.state.playing:
    if st.session_state.recording_state == 'stopped':
        st.session_state.recording_state = 'recording'
        st.write("녹음 중... 5초 동안 노래를 부르세요.")
        start_time = time.time()
        audio_frames = []
        
        while time.time() - start_time < 5:  # 5초 동안 녹음
            try:
                audio_frame = audio_buffer.get(timeout=0.1)
                audio_frames.append(audio_frame)
            except queue.Empty:
                continue
        
        st.session_state.recording_state = 'stopped'
        
        if audio_frames:
            st.session_state.audio_data = np.concatenate(audio_frames, axis=0)
        else:
            st.error("녹음된 오디오가 없습니다. 마이크 권한을 확인하고 다시 시도해주세요.")

if st.session_state.audio_data is not None:
    audio_data = st.session_state.audio_data
    
    # 특징 추출
    features = extract_features(audio_data, sr=48000)
    features = features.reshape(1, -1)
    
    # 예측
    try:
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)
        voice_type = label_encoder.inverse_transform(predicted_label)[0]
        st.write(f"예측된 음역대: {voice_type}")
    except Exception as e:
        st.error(f"예측 중 오류 발생: {str(e)}")
    
    # MFCC 스펙트로그램 표시
    st.write("MFCC 스펙트로그램:")
    fig, ax = plt.subplots()
    mfcc = librosa.feature.mfcc(y=audio_data, sr=48000, n_mfcc=40)
    librosa.display.specshow(mfcc, sr=48000, x_axis='time', ax=ax)
    plt.colorbar()
    st.pyplot(fig)
    
    # 분석 완료 후 상태 초기화
    st.session_state.audio_data = None

st.write("사용 방법: 'START' 버튼을 클릭하고 5초 동안 노래를 부르세요. 녹음이 자동으로 종료되고 분석 결과가 표시됩니다.")
