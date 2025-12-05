import os
import sys
import subprocess
import importlib
import json
import requests # Need to import requests here for the fix to work in the main loop

# Auto-requirements checker
def check_and_install_requirements():
    """Check and install missing requirements automatically"""
    required_packages = {
        "discord": "discord==2.3.2",
        "requests": "requests==2.31.0", 
        "asyncio": "asyncio==3.4.3",
        "dotenv": "python-dotenv==1.0.0",
        "google.generativeai": "google-generativeai==0.3.2",
        "langdetect": "langdetect==1.0.9",
        "emoji": "emoji==2.10.1",
        "colorama": "colorama==0.4.6",
        "psutil": "psutil==5.9.8",
        "schedule": "schedule==1.2.1",
        "fake_useragent": "fake-useragent==1.4.0",
        "dateutil": "python-dateutil==2.8.2"
    }
    
    missing_packages = []
    
    # Check each package
    for package, pip_name in required_packages.items():
        try:
            if package == "dotenv":
                importlib.import_module("dotenv")
            elif package == "google.generativeai":
                importlib.import_module("google.generativeai")
            elif package == "fake_useragent":
                importlib.import_module("fake_useragent")
            elif package == colorama:
                importlib.import_module("colorama") # Fixed: colorama added
            elif package == "psutil":
                importlib.import_module("psutil")
            elif package == "schedule":
                importlib.import_module("schedule")
            elif package == "dateutil":
                importlib.import_module("dateutil")
            else:
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    # Install missing packages
    if missing_packages:
        print("🔍 Checking Discord Bot Requirements...")
        print("=" * 50)
        print(f"📦 Found {len(missing_packages)} missing packages")
        print("🚀 Installing automatically...")
        print("=" * 50)
        
        for package in missing_packages:
            try:
                print(f"📦 Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully!")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
            
        print("🎉 All packages installed! Restarting bot...")
        print("=" * 50)
        return True
    
    return True

# Check requirements before importing
if not check_and_install_requirements():
    print("❌ Failed to install required packages. Exiting...")
    sys.exit(1)

# Now import all packages
import requests
import random
import time
import asyncio
import google.generativeai as palm
import json
from dotenv import load_dotenv
from langdetect import detect
from emoji import emojize
from datetime import datetime

# Load .env file
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Parse unlimited keys from .env (comma/semicolon/newline separated)
_env_keys_raw = os.getenv("GEMINI_API_KEYS", "")
ENV_GEMINI_KEYS = []
for sep in ["\n", ";", ","]:
    parts = []
    if _env_keys_raw:
        for chunk in _env_keys_raw.split(sep):
            val = (chunk or "").strip()
            if val:
                parts.append(val)
    if parts:
        ENV_GEMINI_KEYS = parts
        break

HEADERS = {
    "Authorization": DISCORD_TOKEN,
    "Content-Type": "application/json"
}

# Max Gemini API keys to use
MAX_GEMINI_KEYS = 10

# Gemini key manager for multi-key rotation (placed early so it's available everywhere)
class GeminiKeyManager:
    def __init__(self, initial_key: str = None):
        self.keys = []
        self.index = 0
        self.cooldowns = {}  # key -> available_after_timestamp
        if initial_key:
            self._add_key(initial_key)
        # Load extra keys from env and config if available
        try:
            for k in (ENV_GEMINI_KEYS or []):
                self._add_key(k)
            if 'config_manager' in globals() and hasattr(config_manager, 'config'):
                extra = config_manager.config.get('gemini_api_keys') or []
                for k in extra:
                    self._add_key(k)
        except Exception:
            pass
        if self.keys:
            palm.configure(api_key=self.keys[0])
    
    def _add_key(self, key: str):
        if isinstance(key, str) and len(key) > 10 and key not in self.keys:
            self.keys.append(key)
            self.cooldowns[key] = 0
    
    def current_key(self):
        return self.keys[self.index] if self.keys else None
    
    def _advance_to_available(self):
        if not self.keys:
            return None
        start = self.index
        now = time.time()
        for _ in range(len(self.keys)):
            k = self.keys[self.index]
            if now >= self.cooldowns.get(k, 0):
                palm.configure(api_key=k)
                return k
            self.index = (self.index + 1) % len(self.keys)
        return None
    
    def next_key(self):
        if not self.keys:
            return None
        self.index = (self.index + 1) % len(self.keys)
        return self._advance_to_available()
    
    def set_cooldown(self, key: str, seconds: int):
        if key in self.cooldowns:
            self.cooldowns[key] = max(self.cooldowns.get(key, 0), time.time() + max(0, seconds))

# Initialize key manager early
key_manager = None
try:
    key_manager = GeminiKeyManager(GEMINI_API_KEY)
except Exception:
    key_manager = None

# Check requirements before importing
if not check_and_install_requirements():
    print("❌ Failed to install required packages. Exiting...")
    sys.exit(1)

# Now import all packages
import requests
import random
import time
import asyncio
import google.generativeai as palm
import json
from dotenv import load_dotenv
from langdetect import detect
from emoji import emojize
from datetime import datetime

# Load .env file
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HEADERS = {
    "Authorization": DISCORD_TOKEN,
    "Content-Type": "application/json"
}

# Gemini AI Configuration
palm.configure(api_key=GEMINI_API_KEY)
model = palm.GenerativeModel("gemini-2.5-flash")

# Language Detection Function
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# GET_RANDOM_EMOJIS FUNCTION IS NO LONGER USED, BUT WE KEEP IT FOR COMPATIBILITY
# Get Random Emojis based on message sentiment
def get_random_emojis(count=2, sentiment='happy'):
    emoji_map = {
        'happy': [':grinning_face:', ':beaming_face_with_smiling_eyes:', ':face_with_tears_of_joy:', 
                  ':smiling_face_with_hearts:', ':star-struck:', ':face_blowing_a_kiss:', 
                  ':smiling_face_with_heart-eyes:', ':winking_face:', ':partying_face:'],
        'thinking': [':thinking_face:', ':face_with_monocle:', ':face_with_raised_eyebrow:', 
                     ':face_with_hand_over_mouth:', ':nerd_face:', ':face_with_hand_over_mouth:'],
        'helpful': [':thumbs_up:', ':OK_hand:', ':raising_hands:', ':folded_hands:', 
                    ':sparkles:', ':light_bulb:', ':check_mark:', ':rocket:'],
        'sympathetic': [':slightly_smiling_face:', ':hugging_face:', ':relieved_face:', ':heart:', ':pray:'],
        'confused': [':confused_face:', ':thinking_face:', ':face_with_raised_eyebrow:', ':face_with_monocle:'],
        'excited': [':partying_face:', ':star-struck:', ':face_with_cowboy_hat:', ':fire:', ':zap:'],
        'supportive': [':thumbs_up:', ':heart:', ':pray:', ':clap:', ':muscle:', ':tada:']
    }
    emojis = emoji_map.get(sentiment, emoji_map['happy'])
    selected = random.sample(emojis, min(count, len(emojis)))
    # We return raw string, the main function handles the emoji removal now
    return ' '.join(emojize(emoji) for emoji in selected) 


# Rate limiting for API calls
class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self):
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

# Initialize rate limiter (max 15 requests per minute - safer for Discord)
ai_rate_limiter = RateLimiter(max_requests=15, time_window=60)

# Channel slow mode configuration
CHANNEL_SLOW_MODES = {}

# Anti-ban configuration
ANTI_BAN_CONFIG = {
    'max_continuous_hours': 4,  # Max 4 hours continuously
    'break_duration_minutes': 30,  # 30 minutes break
    'max_daily_hours': 12,  # Max 12 hours per day
    'channel_rotation': True,  # Auto switch channels
    'response_patterns': ['casual', 'professional', 'funny', 'helpful', 'quiet']
}

# AI Reply Function with improved context handling
def get_gemini_response(prompt, detected_lang, message_type='general'):
    try:
        # Ensure key manager exists (lazy init)
        global key_manager
        if 'key_manager' not in globals() or key_manager is None:
            extra = []
            try:
                extra = config_manager.config.get('gemini_api_keys') or []
            except Exception:
                extra = []
            keys = []
            if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
                keys.append(GEMINI_API_KEY)
            for k in extra:
                if isinstance(k, str) and len(k) > 10 and k not in keys:
                    keys.append(k)
            # Apply cap
            keys = keys[:MAX_GEMINI_KEYS]
            
            class _KM:
                def __init__(self, keys_list):
                    self.keys = keys_list or []
                    self.index = 0
                    if self.keys:
                        palm.configure(api_key=self.keys[0])
                def current_key(self):
                    return self.keys[self.index] if self.keys else None
                def next_key(self):
                    if not self.keys:
                        return None
                    self.index = (self.index + 1) % len(self.keys)
                    palm.configure(api_key=self.keys[self.index])
                    return self.current_key()
            key_manager = _KM(keys)

        if not ai_rate_limiter.can_make_request():
            return "Rate limit exceeded. Please try again later."

        # === START MODIFICATION FOR ENGLISH-ONLY AND NO EMOJI (REVISED) ===
        # 1. Force English language instruction and persona
        lang_instructions = {
            'en': 'Reply only in English language with 1-2 friendly sentences. Do not use any emojis in the response.'
        }
        
        # 2. Add the Indian persona instruction here:
        base_identity = "You are a friendly, enthusiastic, and helpful human community member on Discord. You live in a major city in Maharashtra, India. Do not reveal that you are an AI model."
        
        # Set detected_lang to 'en' to use the English-only instruction
        detected_lang = 'en'
        
        # Original templates (we rely on the prompt to override them)
        templates = {
            'general': ['Keep the response natural and conversational.',
                        'Add some personality to the response.',
