import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import librosa.display
import io
from audiorecorder import audiorecorder

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

# 오디오 녹음
st.write("아래 버튼을 클릭하고 5초 동안 노래를 부르세요.")
audio = audiorecorder("녹음 시작", "녹음 중지")

if len(audio) > 0:
    # 녹음된 오디오를 재생
    st.audio(audio.export().read())
    
    # 오디오 데이터 처리
    audio_array = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    sr = audio.frame_rate
    
    # 특징 추출
    features = extract_features(audio_array, sr)
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
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
    plt.colorbar()
    st.pyplot(fig)

st.write("사용 방법: '녹음 시작' 버튼을 클릭하고 노래를 부른 후, '녹음 중지' 버튼을 클릭하세요.")
