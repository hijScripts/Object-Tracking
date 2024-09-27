from gtts import gTTS
import os

# Text to convert to speech
text = "Handsome devil detected. Keep at it you handsome little devil."

# Language (en = English)
tts = gTTS(text=text, lang='en')

# Save the speech to a file
tts.save("handsome-devil.mp3")
