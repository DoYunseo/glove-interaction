# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai",
#     "opencv-python",
#     "pupil-labs-realtime-api",
#     "pydub",
#     "python-dotenv",
# ]
# ///

import asyncio
import base64
import io
import os
from typing import AsyncIterator, Literal

import cv2
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pupil_labs.realtime_api import (
    Device,
    Network,
    receive_gaze_data,
    receive_video_frames,
)
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()


class OpenAICost:
    date_updated = "2024-10-21"
    "Cost $ per 1M token"
    model = {
        "gpt-4o": {
            "input_cost": 2.5,
            "input_cost_cached": 1.25,
            "output_cost": 10,
            "neon_frame_input": 0.001913,
        },
        "tts-1": {"output_cost": 15},
        "tts-1-hd": {"output_cost": 30},
    }

    @classmethod
    def input_cost(cls, model):
        if model in cls.model and "input_cost" in cls.model[model]:
            return cls.model[model]["input_cost"]
        else:
            return None

    @classmethod
    def output_cost(cls, model):
        if model in cls.model and "output_cost" in cls.model[model]:
            return cls.model[model]["output_cost"]
        else:
            return None

    @classmethod
    def frame_cost(cls, model):
        if model in cls.model and "neon_frame_input" in cls.model[model]:
            return cls.model[model]["neon_frame_input"]
        else:
            return None


class Assistant:
    def __init__(self):
        self.device = None
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"  # "gpt-4-turbo"
        self.setup_prompts()
        self.mode = "describe"
        self.running = True
        self.key_actions = {
            ord("a"): lambda: setattr(self, "mode", "describe"),
            ord("s"): lambda: setattr(self, "mode", "dangers"),
            ord("d"): lambda: setattr(self, "mode", "intention"),
            ord("f"): lambda: setattr(self, "mode", "in_detail"),
            32: self.handle_space,
            27: lambda: setattr(self, "running", False),
        }
        self.session_cost = 0
        self.voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = (
            "alloy"
        )
        self.audio_format: Literal["mp3", "opus", "aac", "flac", "pcm", "wav"] = "opus"

    async def initialise_device(self):
        print("Looking for the next best device...")
        async with Network() as network:
            self.device = await network.wait_for_new_device(timeout_seconds=5)
        if self.device is None:
            print("No device found.")
            return
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)

        print(f"Connecting to {self.device}...")

    def setup_prompts(self):
        self.prompts = {
            "base": "You are a visual and communication aid for individuals with visual impairment"
            + "(low vision) or communication difficulties, they are wearing eye-tracking glasses, "
            + "I am sending you an image with a red circle indicating the wearer's gaze, do not "
            + "describe the whole image unless explicitly asked, be succinct, reply in: English",
            "describe": "in couple of words (max. 8) say what the person is looking at.",
            "dangers": "briefly indicate if there is any posing risk for the person in the scene, "
            + "be succinct (max 30 words).",
            "intention": "given that the wearer has mobility and speaking difficulties, "
            + "briefly try to infer the wearer's intention based on what they are looking "
            + "at (maximum of 30 words).",
            "in_detail": "describe the scene in detail, with a maximum duration of one minute of speaking.",
        }

    async def process_frame(self):
        async with Device.from_discovered_device(self.device) as device:
            status = await device.get_status()

            sensor_gaze = status.direct_gaze_sensor()
            if not sensor_gaze.connected:
                print(f"Gaze sensor is not connected to {device}")
                return

            sensor_world = status.direct_world_sensor()
            if not sensor_world.connected:
                print(f"Scene camera is not connected to {device}")
                return

            restart_on_disconnect = True

            queue_video = asyncio.Queue()
            queue_gaze = asyncio.Queue()

            process_video = asyncio.create_task(
                self.enqueue_sensor_data(
                    receive_video_frames(
                        sensor_world.url, run_loop=restart_on_disconnect
                    ),
                    queue_video,
                )
            )
            process_gaze = asyncio.create_task(
                self.enqueue_sensor_data(
                    receive_gaze_data(sensor_gaze.url, run_loop=restart_on_disconnect),
                    queue_gaze,
                )
            )
            try:
                await self.match_and_draw(queue_video, queue_gaze)
            finally:
                process_video.cancel()
                process_gaze.cancel()

    async def enqueue_sensor_data(
        self, sensor: AsyncIterator, queue: asyncio.Queue
    ) -> None:
        async for datum in sensor:
            try:
                queue.put_nowait((datum.datetime, datum))
            except asyncio.QueueFull:
                print(f"Queue is full, dropping {datum}")

    async def get_most_recent_item(self, queue):
        item = await queue.get()
        while True:
            try:
                next_item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return item
            else:
                item = next_item

    async def get_closest_item(self, queue, timestamp):
        item_ts, item = await queue.get()
        # assumes monotonically increasing timestamps
        if item_ts > timestamp:
            return item_ts, item
        while True:
            try:
                next_item_ts, next_item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return item_ts, item
            else:
                if next_item_ts > timestamp:
                    return next_item_ts, next_item
                item_ts, item = next_item_ts, next_item

    async def match_and_draw(self, queue_video, queue_gaze):
        while self.running:
            video_datetime, video_frame = await self.get_most_recent_item(queue_video)
            _, gaze_datum = await self.get_closest_item(queue_gaze, video_datetime)

            self.matched = video_frame.to_ndarray(format="bgr24")

            cv2.circle(
                self.matched,
                (int(gaze_datum.x), int(gaze_datum.y)),
                radius=40,
                color=(0, 0, 255),
                thickness=5,
            )

            self.bgr_pixels = cv2.putText(
                self.matched,
                str(self.mode),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                2,
                cv2.LINE_8,
            )
            cv2.imshow("Scene camera with gaze overlay", self.bgr_pixels)
            key = cv2.waitKey(1) & 0xFF
            if key in self.key_actions:
                self.key_actions[key]()

    async def encode_image(self):
        loop = asyncio.get_event_loop()
        _, buffer = await loop.run_in_executor(None, cv2.imencode, ".jpg", self.matched)
        self.base64Frame = base64.b64encode(buffer).decode("utf-8")

    async def assist(self):
        await self.encode_image()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.prompts["base"] + self.prompts[self.mode],
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
        response_cost = (
            response.usage.prompt_tokens / int(1e6) * OpenAICost.input_cost(self.model)
            + response.usage.completion_tokens
            / int(1e6)
            * OpenAICost.output_cost(self.model)
            + OpenAICost.frame_cost(self.model)
        )
        response_audio = await self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            response_format=self.audio_format,
            input=response.choices[0].message.content,
        )
        TTS_cost = (
            len(response.choices[0].message.content)
            / int(1e6)
            * OpenAICost.output_cost("tts-1")
        )
        self.session_cost += response_cost + TTS_cost
        print(
            f"R: {response.choices[0].message.content}, approx cost(GPT/TTS): {response_cost} / {TTS_cost} $ Total: {response_cost + TTS_cost} $"
        )
        byte_stream = io.BytesIO(response_audio.content)
        audio = AudioSegment.from_file(byte_stream)
        audio = audio.speedup(playback_speed=1.1)
        play(audio)

    def handle_space(self):
        asyncio.create_task(self.assist())

    async def run(self):
        await self.initialise_device()
        while self.device is not None and self.running:
            await self.process_frame()
        print("Stopping...")
        print(f"Total session cost {self.session_cost}$")


if __name__ == "__main__":
    eyes = Assistant()
    asyncio.run(eyes.run())
