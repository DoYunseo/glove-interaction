import base64
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import openai
import os
import torch
import keyboard
import time
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI
from pupil_labs.realtime_api.simple import discover_one_device  # ★ eye tracker

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
stt_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 유틸 --------------------
def timestamped_fname(prefix="base", ext="jpg"):
    t = time.localtime()
    return f"{prefix}_{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}_{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}.{ext}"

def image_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -------------------- Eye tracker 연결 --------------------
def connect_eye_tracker(max_wait_sec=10):
    print("🔎 Eye tracker 탐색 중...")
    device = discover_one_device(max_search_duration_seconds=max_wait_sec)
    if device is None:
        print("❌ Eye tracker를 찾지 못했습니다. 같은 네트워크인지와 Companion 앱 실행 여부를 확인하세요.")
        # 필요시: IP로 직접 연결 가능 (예시)
        # from pupil_labs.realtime_api.simple import Device
        # device = Device(address="192.168.1.169", port=8080)
        # print("IP로 직접 연결 시도:", device)
    else:
        print(f"✅ 연결됨: {device}")
    return device

def capture_base_image_from_et(device):
    """
    Eye tracker의 씬 카메라에서 한 프레임을 받아 파일로 저장.
    output/base/YYYYMMDD_HHMMSS.png
    """
    if device is None:
        print("❌ eye tracker 미연결 상태입니다.")
        return None

    try:
        bgr_pixels, frame_datetime = device.receive_scene_video_frame()  # Pupil Labs API
    except Exception as e:
        print("❌ 씬 카메라 프레임 수신 실패:", e)
        return None

    if bgr_pixels is None:
        print("❌ 빈 프레임")
        return None

    # 저장 폴더 준비
    os.makedirs("output/base", exist_ok=True)

    # 파일명: 년월일_시분초.jpg
    t = time.localtime()
    fname = f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}_{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}.jpg"
    full_path = os.path.join("output/base", fname)

    cv2.imwrite(full_path, bgr_pixels)
    print(f"📷 베이스 이미지 저장: {full_path} (timestamp: {frame_datetime})")
    return full_path


# -------------------- 공압장갑 제어 --------------------
def trigger_glove(pose: str):
    if pose == "thumbs_up":
        print("🧤 공압 장갑: 위로 따봉 포즈 실행")
    elif pose == "fist":
        print("🧤 공압 장갑: 주먹 포즈 실행")
    else:
        print("❌ 알 수 없는 포즈")

# -------------------- 녹음 + STT --------------------
def record_and_transcribe_with_spacebar(sample_rate=16000, mic_device=1, et_device=None):
    print("스페이스바를 눌러 녹음을 시작하고, 다시 누르면 종료합니다.")
    keyboard.wait('space')  # 시작 키 입력 대기 (키다운 이벤트)
    time.sleep(0.02)        # 바운스 방지

    # 스페이스바를 누른 '그 순간'의 베이스 이미지 캡처 (eye tracker)
    base_img_path = capture_base_image_from_et(et_device)

    # 스페이스바에서 손을 뗄 때까지 잠깐 대기 (즉시 종료 방지)
    while keyboard.is_pressed('space'):
        pass

    print("🎤 녹음 시작...")
    audio = []
    block_size = 1024
    stream = sd.InputStream(samplerate=sample_rate, channels=1, device=mic_device)
    stream.start()

    frame = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.namedWindow("Voice Meter")

    while True:
        block, _ = stream.read(block_size)
        audio.append(block)

        volume = int(np.linalg.norm(block) * 500)
        volume = min(volume, 500)

        frame[:] = 0
        cv2.rectangle(frame, (50, 250), (50 + volume, 270), (0, 255, 0), -1)
        cv2.putText(frame, "Voice Input Volume", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.imshow("Voice Meter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("녹음 취소됨")
            stream.stop()
            cv2.destroyWindow("Voice Meter")
            return None, None, None

        if keyboard.is_pressed('space'):
            break

    stream.stop()
    cv2.destroyWindow("Voice Meter")

    audio_np = np.concatenate(audio, axis=0)
    wav_path = "temp.wav"
    sf.write(wav_path, audio_np, sample_rate)
    segments, _ = stt_model.transcribe(wav_path)
    text = " ".join([seg.text for seg in segments]).strip()
    print("📄 인식된 음성:", text if text else "(빈 텍스트)")
    return text, wav_path, base_img_path

# -------------------- VLM 쿼리 --------------------
def query_vlm_with_image_and_text(text, image_path):
    """
    출력은 반드시 한국어 '예' 또는 '아니요' 한 단어만.
    이미지(eye tracker 씬 프레임) + STT 텍스트를 함께 전달.
    """
    if image_path is None:
        messages = [
            {"role": "system",
             "content": "너는 사용자의 의도를 판단하는 VLM이야. 한국어로 '예' 또는 '아니요' 한 단어만 답해."},
            {"role": "user",
             "content": f"다음 발화가 긍정인지 판단해서 '예' 또는 '아니요' 중 하나만 반환해.\n발화: {text}"}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=5,
            temperature=0
        )
        return resp.choices[0].message.content.strip()

    img_b64 = image_to_b64(image_path)
    user_content = [
        {"type": "text",
         "text": (
             "다음 정보를 함께 고려해서 사용자의 의도가 긍정인지 판별하라.\n"
             "반드시 한국어로 '예' 또는 '아니요' 한 단어만 답해.\n\n"
             f"음성 전사 텍스트: {text if text else '(비어 있음)'}"
         )},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
    ]
    messages = [
        {"role": "system",
         "content": "너는 사용자 의도를 판단하는 VLM이야. 출력은 '예' 또는 '아니요'만 허용된다."},
        {"role": "user", "content": user_content}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=5,
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# -------------------- 응답→포즈 매핑 --------------------
def interpret_pose_kor(response_text):
    # '예'면 thumbs_up, 그 외 전부 fist
    norm = response_text.strip().lower()
    if norm in {"예", "네"} or "예" in response_text or "yes" in response_text:
        return "thumbs_up"
    return "fist"

# -------------------- 메인 --------------------
def main():
    et_device = connect_eye_tracker(max_wait_sec=10)
    if et_device is None:
        print("❌ Eye tracker 연결 실패. 프로그램을 종료합니다.")
        return

    try:
        while True:
            print("\n[1] 마이크로 말하기 (이미지+음성)\n[2] 텍스트 직접 입력 (이미지+텍스트)\n[q] 종료")
            choice = input("👉 입력 방식을 선택하세요: ").strip().lower()

            if choice == "q":
                print("👋 종료합니다.")
                break

            if choice == "1":
                user_text, wav_path, base_img_path = record_and_transcribe_with_spacebar(
                    sample_rate=16000, mic_device=1, et_device=et_device
                )
                if user_text is None and base_img_path is None:
                    continue
            elif choice == "2":
                base_img_path = capture_base_image_from_et(et_device)
                user_text = input("✍️ 텍스트를 입력하세요: ").strip()
                if not user_text and base_img_path is None:
                    print("❌ 입력이 비어 있고 이미지도 없음.")
                    continue
            else:
                print("❗ 유효한 옵션을 선택하세요.")
                continue

            vlm_resp = query_vlm_with_image_and_text(user_text, base_img_path)
            print("🧠 VLM 응답(원문):", vlm_resp)

            pose = interpret_pose_kor(vlm_resp)
            trigger_glove(pose)
    finally:
        print("🔌 Eye tracker 연결 종료")
        try:
            et_device.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
