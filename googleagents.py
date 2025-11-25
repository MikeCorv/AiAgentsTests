import os
import google.generativeai as genai

try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    print("Authentication error!")

