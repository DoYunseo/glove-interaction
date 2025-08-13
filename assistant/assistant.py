import os
import base64
import io
import cv2
import torch
import urllib.request
import numpy as np
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from pupil_labs.realtime_api.simple import discover_one_device
from segment_anything import sam_model_registry, SamPredictor

load_dotenv()

# SAM 모델 준비
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


class Assistant:
    def __init__(self):
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.mode = "describe"
        self.running = True
        self.session_cost = 0
        self.key_actions = {
            ord("a"): lambda: setattr(self, "mode", "describe"),
            ord("s"): lambda: setattr(self, "mode", "dangers"),
            ord("d"): lambda: setattr(self, "mode", "intention"),
            ord("f"): lambda: setattr(self, "mode", "in_detail"),
            32: self.handle_space,
            ord("q"): lambda: setattr(self, "running", False),
        }
        self.prompts = {
            "base": "You are a visual and communication aid for individuals with visual impairment"
            + "(low vision) or communication difficulties. They are wearing eye-tracking glasses. "
            + "The red circle marks gaze. Do not describe entire image unless asked. Be succinct.",
            "describe": "in couple of words (max. 8) say what the person is looking at.",
            "dangers": "briefly indicate if there is any posing risk for the person in the scene, be succinct (max 30 words).",
            "intention": "given that the wearer has mobility and speaking difficulties, briefly infer the wearer's intention based on the gaze (max 30 words).",
            "in_detail": "describe the scene in detail, with a maximum duration of one minute of speaking.",
        }
        os.makedirs("output/base", exist_ok=True)
        os.makedirs("output/crop", exist_ok=True)

    def process_frame(self):
        self.matched = self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        if not self.matched:
            return
        self.annotate_and_show_frame()

    def annotate_and_show_frame(self):
        self.frame_with_overlay = self.matched.scene.bgr_pixels.copy()
        cv2.circle(
            self.frame_with_overlay,
            (int(self.matched.gaze.x), int(self.matched.gaze.y)),
            radius=40,
            color=(0, 0, 255),
            thickness=5,
        )
        cv2.putText(
            self.frame_with_overlay,
            str(self.mode),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
            cv2.LINE_8,
        )
        cv2.imshow("Scene", self.frame_with_overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in self.key_actions:
            self.key_actions[key]()

    def encode_image(self):
        _, buffer = cv2.imencode(".jpg", self.frame_with_overlay)
        self.base64Frame = base64.b64encode(buffer).decode("utf-8")

    def save_images(self):
        filename = f"frame_{cv2.getTickCount()}.jpg"
        base_path = os.path.join("output", "base", filename)
        crop_path = os.path.join("output", "crop", filename)

        # base 이미지 저장 (gaze 원 포함)
        cv2.imwrite(base_path, self.frame_with_overlay)

        # crop 이미지 저장 (gaze 원 없이, SAM crop)
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
        else:
            print("SAM failed to segment. Crop not saved.")

    def assist(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompts["base"] + self.prompts[self.mode],
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        "Here goes the image",
                        {"image": self.base64Frame, "resize": 768},
                    ],
                },
            ],
            max_tokens=200,
        )
        print("R:", response.choices[0].message.content)

    def handle_space(self):
        self.encode_image()
        self.assist()
        self.save_images()

    def run(self):
        while self.device is not None and self.running:
            self.process_frame()
        self.device.close()
        print("Session ended.")


if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
