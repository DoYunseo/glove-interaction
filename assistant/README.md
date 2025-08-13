## Install it and run it:

1. Set up your Python environment and API key using [OpenAI's quick start guide](https://platform.openai.com/docs/quickstart/account-setup), create a .env file alike the example.
2. Clone this gist or download it
3. `pip install -r requirements.txt`
4. `python assistant.py`

5. Alternatively, if using [astral/uv](https://astral.sh/), simply `uv run assistant.py`

## Using it

If you have your computer and Companion Device (Neon or Pupil Invisible) connected to the same network, it will be automatically linked and start streaming the scene camera with the gaze circle overlay.

Press **Space** to send the snap to GPT-4 and you will get the response by voice, or use “ASDF” keys to change its model.

**A** - Describe briefly the object gazed.

**S** - Describe any potential danger, knife, roads, …

**D** - Try to guess the wearers intention, wants to drink water, make a call, be moved somewhere…

**F** - More detailed description of the environment.

Press **ESC** to stop the application.
