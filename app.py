import asyncio
import datetime
import hashlib
import json
import logging
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import edge_tts
import streamlit as st
from edge_tts.exceptions import EdgeTTSException
from groq import APIError, Groq
from streamlit.errors import StreamlitSecretNotFoundError


st.set_page_config(
    page_title="집엔 스피커", 
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """너는 '집엔 스피커'의 AI 음성 비서이다.
시니어와 디지털 약자를 위해 핵심부터 쉽고 친절하게 대답해라.
전문 용어와 복잡한 문장을 피하고 짧고 명확하게 말해라.
모르는 내용은 추측하지 말고 솔직히 모른다고 말해라.
답변 마지막에는 항상 "말씀하실 내용이 더 있으신가요?"라고 물어라."""
VOICES = {
    "여성": "ko-KR-SunHiNeural",
    "남성": "ko-KR-InJoonNeural",
}
WEATHER_LABELS = {
    0: "맑음",
    1: "대체로 맑음",
    2: "구름 조금",
    3: "흐림",
    45: "안개",
    48: "안개",
    51: "이슬비",
    53: "이슬비",
    55: "이슬비",
    61: "비",
    63: "비",
    65: "강한 비",
    71: "눈",
    73: "눈",
    75: "강한 눈",
    80: "소나기",
    81: "소나기",
    82: "강한 소나기",
    95: "천둥번개",
}
TIMEOUT_PROMPT = "말씀하세요. 듣고 있어요."
TIMEOUT_STANDBY = "필요할 때에 다시 말씀하세요."


def initialize_state() -> None:
    defaults = {
        "turn_count": 0,
        "voice_gender": "여성",
        "voice_speed": "+0%",
        "app_state": "IDLE",
        "listening_started_at": None,
        "prompted": False,
        "conversation_history": [],
        "question_text": "",
        "response_text": "",
        "response_audio": b"",
        "prompt_audio": b"",
        "tts_error": "",
        "standby_audio": b"",
        "welcome_audio": b"",
        "weather": None,
        "weather_error": None,
        "error_message": "",
        "audio_sequence": 0,
        "last_processed_audio": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data(ttl=1800, show_spinner=False)
def get_jeju_weather() -> tuple[str | None, str | None]:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=33.4996&longitude=126.5312"
        "&current=temperature_2m,weather_code&timezone=Asia%2FSeoul"
    )
    request = Request(url, headers={"User-Agent": "JibeonSpeaker/1.0"})
    try:
        with urlopen(request, timeout=4) as response:
            data = json.loads(response.read())
        current = data["current"]
        weather_code = int(current["weather_code"])
        temperature = round(float(current["temperature_2m"]))
    except (URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        LOGGER.warning("Could not retrieve Jeju weather: %s", error)
        return None, "현재 날씨 정보를 불러오지 못했습니다."

    label = WEATHER_LABELS.get(weather_code, "날씨 정보")
    return f"{label}, 현재 {temperature}도", None


async def _collect_speech(text: str, voice: str, rate: str) -> bytes:
    communicator = edge_tts.Communicate(text, voice, rate=rate)
    chunks = [
        chunk["data"]
        async for chunk in communicator.stream()
        if chunk["type"] == "audio"
    ]
    return b"".join(chunks)


def synthesize_speech(text: str) -> bytes:
    try:
        audio = asyncio.run(
            asyncio.wait_for(
                _collect_speech(
                    text,
                    VOICES[st.session_state.voice_gender],
                    st.session_state.voice_speed,
                ),
                timeout=8,
            )
        )
    except (EdgeTTSException, OSError, TimeoutError) as error:
        LOGGER.warning("Speech synthesis failed: %s", error)
        st.session_state.tts_error = "음성 출력을 준비하지 못했습니다. 화면 자막을 확인해 주세요."
        return b""
    if not audio:
        st.session_state.tts_error = "음성 출력을 준비하지 못했습니다. 화면 자막을 확인해 주세요."
        return b""
    st.session_state.tts_error = ""
    return audio


def get_api_key() -> str:
    try:
        configured_key = st.secrets.get("GROQ_API_KEY", "")
    except StreamlitSecretNotFoundError:
        configured_key = ""
    return (
        configured_key.strip()
        or os.environ.get("GROQ_API_KEY", "").strip()
        or st.session_state.groq_api_key_input.strip()
    )


def process_recording(audio_bytes: bytes) -> None:
    api_key = get_api_key()
    if not api_key:
        st.session_state.error_message = (
            "Groq API 키가 없습니다. 설정에서 키를 입력하거나 "
            "GROQ_API_KEY 환경 변수 또는 Streamlit secrets에 등록해 주세요."
        )
        st.session_state.app_state = "ERROR"
        return

    client = Groq(api_key=api_key, timeout=20.0)
    transcription = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes, "audio/wav"),
        model="whisper-large-v3-turbo",
        language="ko",
    )
    question = transcription.text.strip()
    if not question:
        st.session_state.error_message = "음성을 인식하지 못했습니다. 다시 말씀해 주세요."
        st.session_state.app_state = "ERROR"
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(st.session_state.conversation_history)
    messages.append({"role": "user", "content": question})
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=300,
    )
    answer = completion.choices[0].message.content
    if not answer:
        st.session_state.error_message = "답변을 받지 못했습니다. 다시 시도해 주세요."
        st.session_state.app_state = "ERROR"
        return

    answer = answer.strip()
    st.session_state.question_text = question
    st.session_state.response_text = answer
    st.session_state.response_audio = synthesize_speech(answer)
    st.session_state.conversation_history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    st.session_state.turn_count += 1
    st.session_state.app_state = "RESPONDING"


def start_listening() -> None:
    st.session_state.audio_sequence += 1
    st.session_state.listening_started_at = time.monotonic()
    st.session_state.prompted = False
    st.session_state.prompt_audio = b""
    st.session_state.standby_audio = b""
    st.session_state.error_message = ""
    st.session_state.tts_error = ""
    st.session_state.last_processed_audio = ""
    st.session_state.app_state = "LISTENING"


@st.fragment(run_every=1)
def render_listening_panel() -> None:
    started_at = st.session_state.listening_started_at
    if started_at is None:
        return

    elapsed = time.monotonic() - started_at
    if elapsed >= 10:
        st.session_state.standby_audio = synthesize_speech(TIMEOUT_STANDBY)
        st.session_state.listening_started_at = None
        st.session_state.app_state = "STANDBY"
        st.rerun()

    if elapsed >= 3 and not st.session_state.prompted:
        st.session_state.prompted = True
        st.session_state.prompt_audio = synthesize_speech(TIMEOUT_PROMPT)
        st.rerun()

    if st.session_state.prompted:
        st.markdown('<div class="caption">말씀하세요.</div>', unsafe_allow_html=True)
        st.caption("듣고 있어요.")
        if st.session_state.tts_error:
            st.warning(st.session_state.tts_error)
    else:
        st.markdown('<div class="caption">듣고 있어요.</div>', unsafe_allow_html=True)

    recording = st.audio_input(
        "마이크를 눌러 말씀하시고, 녹음을 마치면 정지해 주세요.",
        key=f"audio_input_{st.session_state.audio_sequence}",
    )
    if recording is None:
        return

    audio_bytes = recording.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if audio_hash == st.session_state.last_processed_audio:
        return

    st.session_state.last_processed_audio = audio_hash
    st.session_state.app_state = "RESPONDING"
    with st.spinner("말씀을 듣고 답변을 준비하고 있어요."):
        try:
            process_recording(audio_bytes)
        except APIError:
            LOGGER.exception("Groq request failed")
            st.session_state.error_message = (
                "AI 서비스에 연결하지 못했습니다. API 키와 네트워크를 확인해 주세요."
            )
            st.session_state.app_state = "ERROR"
    st.rerun()


initialize_state()
seoul_timezone = datetime.timezone(datetime.timedelta(hours=9))
now = datetime.datetime.now(seoul_timezone)
is_morning = 6 <= now.hour < 10

st.markdown(
    """
    <style>
    .caption {
        font-size: clamp(2rem, 6vw, 3.5rem);
        font-weight: 700;
        line-height: 1.35;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton > button, .stAudioInput button {
        min-height: 3.5rem;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .stMarkdown h3 {
        font-size: clamp(2rem, 6vw, 3.5rem);
        line-height: 1.35;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

header, settings = st.columns([8, 2])
with header:
    if st.session_state.turn_count >= 5:
        st.info("📢 [스폰서 광고] 0원 AI 스피커 제휴 서비스 안내")
with settings:
    with st.popover("⚙️ 설정"):
        st.radio("음성 성별", list(VOICES), key="voice_gender")
        st.select_slider(
            "음성 속도",
            options=["-10%", "+0%", "+10%"],
            key="voice_speed",
        )
        st.text_input(
            "Groq API 키",
            type="password",
            key="groq_api_key_input",
            help="키는 이 브라우저 세션에서만 사용됩니다.",
        )
        st.markdown("[제휴 요금제 및 서비스 안내](#)")

st.title("집엔 스피커")
st.caption("편하게 말씀하시면 쉬운 말로 답해 드릴게요.")
if st.session_state.tts_error:
    st.warning(st.session_state.tts_error)

if is_morning:
    if st.session_state.weather is None and st.session_state.weather_error is None:
        weather, weather_error = get_jeju_weather()
        st.session_state.weather = weather
        st.session_state.weather_error = weather_error

    if st.session_state.weather:
        greeting = (
            f"안녕하세요? 오늘 제주 날씨는 {st.session_state.weather}입니다. "
            "무엇을 도와드릴까요?"
        )
        st.info(greeting)
        if st.button("🔊 아침 인사 듣기", key="morning_greeting_button"):
            st.session_state.welcome_audio = synthesize_speech(greeting)
        if st.session_state.welcome_audio:
            st.audio(
                st.session_state.welcome_audio,
                format="audio/mp3",
                autoplay=True,
            )
    else:
        st.info("안녕하세요? 무엇을 도와드릴까요?")
        st.warning(st.session_state.weather_error)

if (
    st.session_state.app_state == "LISTENING"
    and st.session_state.prompted
    and st.session_state.prompt_audio
):
    st.audio(
        st.session_state.prompt_audio,
        format="audio/mp3",
        autoplay=True,
    )

if st.session_state.app_state == "LISTENING":
    render_listening_panel()
elif st.session_state.app_state == "RESPONDING":
    st.markdown(f"### {st.session_state.response_text}")
    if st.session_state.question_text:
        st.caption(f"질문: {st.session_state.question_text}")
    if st.session_state.response_audio:
        st.audio(st.session_state.response_audio, format="audio/mp3", autoplay=True)
    if st.button("🎙️ 이어서 대화하기", type="primary"):
        start_listening()
        st.rerun()
elif st.session_state.app_state == "STANDBY":
    st.markdown(f'<div class="caption">{TIMEOUT_STANDBY}</div>', unsafe_allow_html=True)
    if st.session_state.standby_audio:
        st.audio(st.session_state.standby_audio, format="audio/mp3", autoplay=True)
    if st.button("🎙️ 다시 시작하기", type="primary"):
        start_listening()
        st.rerun()
elif st.session_state.app_state == "ERROR":
    st.error(st.session_state.error_message)
    if st.button("다시 시도하기", type="primary"):
        st.session_state.app_state = "IDLE"
        st.rerun()
else:
    st.markdown('<div class="caption">무엇을 도와드릴까요?</div>', unsafe_allow_html=True)
    if not get_api_key():
        st.warning(
            "대화를 시작하려면 설정에서 Groq API 키를 입력하거나 "
            "GROQ_API_KEY 환경 변수 또는 Streamlit secrets에 등록해 주세요."
        )
    if st.button("🎙️ 마이크로 대화 시작", type="primary"):
        start_listening()
        st.rerun()
