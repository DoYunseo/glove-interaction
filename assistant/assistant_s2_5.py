import os
import base64
import cv2
import time
import torch
import urllib.request
import numpy as np
from collections import deque
from dotenv import load_dotenv
from openai import OpenAI
from pupil_labs.realtime_api.simple import discover_one_device
from segment_anything import sam_model_registry, SamPredictor

# NEW: hand detection
import mediapipe as mp
mp_hands = mp.solutions.hands

# -------------------- NEW: tunables --------------------
PROX_THRESH_PX = 50          # "충분히 가까움" 판정 임계값 (px)
BOTTOM_BAND_FRAC = 0.18      # 바닥 띠 두께(바운딩박스 높이의 비율)
OPEN_KERNEL_FRAC = 0.03      # opening 커널 크기(영상 짧은 변 대비 비율)
# ------------------------------------------------------

# Load environment variables
load_dotenv()

# Load SAM model
SAM_CHECKPOINT_PATH = "sam_vit_h.pth"
SAM_MODEL_TYPE = "vit_h"

if not os.path.exists(SAM_CHECKPOINT_PATH):
    print("Downloading SAM checkpoint...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        SAM_CHECKPOINT_PATH,
    )

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)


# -------------------- NEW: helpers --------------------
def _ensure_binary(mask):
    m = (mask > 0).astype(np.uint8) * 255
    return m

def _largest_cc(mask):
    mask = _ensure_binary(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == idx).astype(np.uint8) * 255
    return out

def _odd(n):
    return int(n) if int(n) % 2 == 1 else int(n) + 1

def split_cup_parts(obj_mask):
    """
    obj_mask (uint8 0/255) -> dict{'cup_1','cup_2','cup_3'} 마스크(0/255)
    cup_1: 벽(body minus bottom)
    cup_2: 손잡이(handle)
    cup_3: 바닥(bottom strip)
    """
    m = _ensure_binary(obj_mask)
    h, w = m.shape[:2]
    if cv2.countNonZero(m) == 0:
        return {'cup_1': np.zeros_like(m), 'cup_2': np.zeros_like(m), 'cup_3': np.zeros_like(m)}

    # bounding box
    ys, xs = np.where(m > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)

    # opening으로 얇은 구조(손잡이) 제거 → 본체(body) 근사
    k = max(3, int(min(h, w) * OPEN_KERNEL_FRAC))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    opened = _largest_cc(opened)  # 본체만 남김

    # 손잡이 후보: 원본 - opened
    handle_cand = cv2.subtract(m, opened)

    # 손잡이 노이즈 억제: 좌/우 측면 쪽만 유지 (중앙부 제외)
    center_x = (x0 + x1) / 2.0
    side_band = int(bw * 0.22)  # 양쪽 22% 스트립만
    side_mask = np.zeros_like(m)
    side_mask[y0:y1+1, max(x0, x0):min(x0+side_band, x1+1)] = 255  # left
    side_mask[y0:y1+1, max(x1-side_band+1, x0):x1+1] = 255         # right
    handle_cand = cv2.bitwise_and(handle_cand, side_mask)

    # 가장 큰 연결성분만 손잡이로
    handle = _largest_cc(handle_cand)

    # 바닥: opened 하단 band
    band_h = max(3, int(bh * BOTTOM_BAND_FRAC))
    bottom_band = np.zeros_like(m)
    yb0 = min(y1 - band_h + 1, y1)
    bottom_band[yb0:y1+1, x0:x1+1] = 255
    bottom = cv2.bitwise_and(opened, bottom_band)

    # 벽: opened - bottom
    wall = cv2.subtract(opened, bottom)

    return {'cup_1': wall, 'cup_2': handle, 'cup_3': bottom}

def min_distance_region_to_hand(region_mask, hand_mask):
    """
    region_mask: 0/255, hand_mask: 0/255
    반환: 최소거리(px, float). 영역 겹치면 0.
    """
    if cv2.countNonZero(region_mask) == 0 or cv2.countNonZero(hand_mask) == 0:
        return np.inf
    # 영역을 0으로, 배경 255로 → distanceTransform으로 "영역까지" 거리
    src = np.where(region_mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(src, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    ys, xs = np.where(hand_mask > 0)
    vals = dist[ys, xs]
    return float(vals.min()) if vals.size > 0 else np.inf
# -----------------------------------------------------


class Assistant:
    def __init__(self):
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

        self.gaze_history = deque(maxlen=90)  # Assuming 30 FPS, 3 seconds
        self.last_trigger_time = 0
        self.running = True

        # UPDATED: 최근 물체 마스크/시각 + 거리 출력 주기
        self.last_obj_mask = None
        self.last_seg_time = 0.0
        self.last_distance_print_time = 0.0

        # MediaPipe Hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        os.makedirs("output/base", exist_ok=True)
        os.makedirs("output/crop", exist_ok=True)

        self.prompts = {
            "base": "You are a visual and communication aid for individuals"
            + "They are wearing eye-tracking glasses. "
            + "The red circle marks gaze. Do not describe entire image unless asked. Be succinct. ",
            "describe": "In a couple of words (max. 8), say what the person is looking at."
        }

    def run(self):
        while self.device is not None and self.running:
            self.process_frame()
        self.device.close()
        print("Session ended.")

    def process_frame(self):
        matched = self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        if not matched:
            return

        self.matched = matched
        raw_frame = matched.scene.bgr_pixels.copy()
        self.frame_with_overlay = raw_frame.copy()
        gaze_point = (int(matched.gaze.x), int(matched.gaze.y))

        # gaze 표시 + quit 힌트
        cv2.circle(self.frame_with_overlay, gaze_point, radius=40, color=(0, 0, 255), thickness=5)
        cv2.putText(self.frame_with_overlay, "Press 'q' to quit", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 1초마다 손-물체 최소거리 & 포즈 라벨 출력
        self.maybe_print_min_distance_and_pose(raw_frame, gaze_point)

        cv2.imshow("Scene", self.frame_with_overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
            return

        # fixation 판정용 이력
        self.gaze_history.append((matched.gaze.x, matched.gaze.y, time.time()))
        self.check_gaze_fixation()

    def compute_hand_mask(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0].landmark
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm], dtype=np.int32)
        hull = cv2.convexHull(pts)
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hand_mask, hull, 255)

        # 시각화
        cv2.polylines(self.frame_with_overlay, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
        return hand_mask

    # -------------------- UPDATED: 핵심 함수 --------------------
    def maybe_print_min_distance_and_pose(self, frame_bgr, gaze_point):
        now = time.time()
        if now - self.last_distance_print_time < 1.0:
            return
        self.last_distance_print_time = now

        hand_mask = self.compute_hand_mask(frame_bgr)
        if hand_mask is None or not np.any(hand_mask):
            print("[DIST] Hand not detected in frame.")
            return

        # SAM 물체 마스크가 있으면 이를 파츠로 분해 → 각 파츠와 손 사이 최소거리
        if self.last_obj_mask is not None and (now - self.last_seg_time) < 3.5:
            obj_mask = self.last_obj_mask
            if obj_mask.shape != hand_mask.shape:
                obj_mask = cv2.resize(obj_mask, (hand_mask.shape[1], hand_mask.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            parts = split_cup_parts(obj_mask)  # {'cup_1','cup_2','cup_3'} → 0/255 mask
            dists = {}
            for label, pmask in parts.items():
                d = min_distance_region_to_hand(pmask, hand_mask)
                dists[label] = d

            # 전체 컵에 대한 최소거리도 계산(참고용)
            overall_d = min_distance_region_to_hand(obj_mask, hand_mask)

            # 가장 가까운 파츠와 거리
            best_label = min(dists, key=lambda k: dists[k]) if dists else None
            best_dist = dists.get(best_label, np.inf)

            # 충분히 가까우면 라벨 출력
            pose_out = best_label if best_dist < PROX_THRESH_PX else "-"

            # 콘솔 출력
            # 예) [DIST] min=12.3 px | pose: cup_2
            print(f"[DIST] Hand↔Object min: {overall_d:.1f} px | "
                  f"cup_1={dists['cup_1']:.1f}, cup_2={dists['cup_2']:.1f}, cup_3={dists['cup_3']:.1f} | "
                  f"pose: {pose_out}")

            # (선택) 프레임에 라벨도 살짝 그려줌
            cv2.putText(self.frame_with_overlay, f"pose: {pose_out}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        else:
            # 아직 컵 마스크가 없으면 fallback: 시선점과 손 사이 근사거리만
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                print("[DIST] Hand mask empty.")
                return
            cnt = max(contours, key=cv2.contourArea)
            d = cv2.pointPolygonTest(cnt, gaze_point, True)
            min_dist = max(0.0, -float(d))
            print(f"[DIST] Hand↔Gaze-proxy (no SAM yet): {min_dist:.1f} px | pose: -")
    # -----------------------------------------------------------

    def check_gaze_fixation(self):
        if len(self.gaze_history) < self.gaze_history.maxlen:
            return

        xs, ys = zip(*[(x, y) for x, y, _ in self.gaze_history])
        x_std, y_std = np.std(xs), np.std(ys)

        if x_std < 20 and y_std < 20:
            if time.time() - self.last_trigger_time > 5:
                self.last_trigger_time = time.time()
                print("\n[INFO] Gaze fixation detected. Running LLM + SAM...")
                self.handle_gaze_trigger()

    def handle_gaze_trigger(self):
        self.encode_image()
        self.assist()
        self.save_images()

    def encode_image(self):
        _, buffer = cv2.imencode(".jpg", self.frame_with_overlay)
        self.base64Frame = base64.b64encode(buffer).decode("utf-8")

    def assist(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.prompts["base"] + self.prompts["describe"]}],
                },
                {
                    "role": "user",
                    "content": ["Here goes the image", {"image": self.base64Frame, "resize": 768}],
                },
            ],
            max_tokens=200,
        )
        print("\n[LLM Response]:", response.choices[0].message.content)

    def save_images(self):
        # 원하는 형식: frame_YYYYmmdd_HHMMSS.jpg
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"frame_{ts}.jpg"

        base_path = os.path.join("output", "base", filename)
        crop_path = os.path.join("output", "crop", filename)

        # base 저장
        cv2.imwrite(base_path, self.frame_with_overlay)

        # SAM crop + 마스크 저장
        raw_frame = self.matched.scene.bgr_pixels.copy()
        gaze_point = np.array([[self.matched.gaze.x, self.matched.gaze.y]])
        predictor.set_image(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
        masks, _, _ = predictor.predict(
            point_coords=gaze_point,
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0]
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            crop = raw_frame[y_min:y_max, x_min:x_max]
            cv2.imwrite(crop_path, crop)

            # 최근 컵 마스크 업데이트
            self.last_obj_mask = (mask.astype(np.uint8) * 255)
            self.last_seg_time = time.time()
        else:
            print("[WARN] SAM failed to segment. Crop not saved.")
            self.last_obj_mask = None



if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
