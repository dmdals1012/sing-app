import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model

# 모델 로드
@st.cache_resource
def load_model():
    return load_model('vocal_range_classifier.keras')


model = load_model()

# 레이블 인코더 로드 (저장했다고 가정)
label_encoder = joblib.load('label_encoder.pkl')

st.title('실시간 음역대 분류기')

# 녹음 함수
def record_audio(duration, fs):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording

# 음성 특징 추출
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio.flatten(), sr=sr, n_mfcc=40)
    mfcc = mfcc.T
    return mfcc

# 녹음 버튼
if st.button('녹음 시작 (5초)'):
    fs = 22050  # 샘플링 레이트
    duration = 5  # 녹음 시간 (초)
    
    st.text('녹음 중...')
    audio = record_audio(duration, fs)
    st.text('녹음 완료!')
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        wav.write(tmpfile.name, fs, audio)
        st.audio(tmpfile.name)
    
    # 특징 추출
    features = extract_features(audio, fs)
    
    # 예측
    prediction = model.predict(features.reshape(1, -1))
    voice_type = label_encoder.inverse_transform(prediction)[0]
    
    st.write(f"예측된 음역대: {voice_type}")

    # MFCC 스펙트로그램 표시
    st.write("MFCC 스펙트로그램:")
    fig, ax = plt.subplots()
    librosa.display.specshow(features.T, x_axis='time', ax=ax)
    st.pyplot(fig)

st.write("사용 방법: '녹음 시작' 버튼을 클릭하고 5초 동안 노래를 부르세요.")


