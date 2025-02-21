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

# 모델 로드
@st.cache_resource
def load_model():
    return keras.models.load_model('vocal_range_classifier.h5')

model = load_model()

# 레이블 인코더 로드
label_encoder = joblib.load('label_encoder.pkl')

st.title('실시간 음역대 분류기')

# 음성 특징 추출
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)  # 평균값으로 고정된 크기의 벡터 생성
    return mfcc

# 오디오 처리 함수
def process_audio(frame):
    sound = frame.to_ndarray()
    sound = sound.astype(np.float32) / 32768.0
    return sound

# 웹 오디오 스트리밍
audio_buffer = queue.Queue()

def audio_receiver(frame):
    sound = process_audio(frame)
    audio_buffer.put(sound)

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.state.playing:
    st.write("녹음 중... 5초 동안 노래를 부르세요.")
    audio_frames = []
    for _ in range(5 * 48):  # 5초 동안 녹음 (48kHz 샘플링 레이트 가정)
        try:
            audio_frame = audio_buffer.get(timeout=1.0)
            audio_frames.append(audio_frame)
        except queue.Empty:
            break
    
    if len(audio_frames) > 0:
        audio_data = np.concatenate(audio_frames, axis=0)
        
        # 특징 추출
        features = extract_features(audio_data, sr=48000)
        features = features.reshape(1, -1)  # 모델 입력 형태 맞춤
        
        # 예측
        try:
            prediction = model.predict(features)
            predicted_label = np.argmax(prediction, axis=1)  # 가장 높은 확률의 클래스를 선택
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

st.write("사용 방법: 'START' 버튼을 클릭하고 5초 동안 노래를 부르세요. 그 후 'STOP' 버튼을 클릭하세요.")
