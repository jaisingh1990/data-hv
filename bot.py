import os
import sys
import subprocess
import importlib
import json
import requests
import random
import time
import asyncio
import google.generativeai as palm
from dotenv import load_dotenv
from langdetect import detect
from emoji import emojize
from datetime import datetime
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows
init()

# --- Configuration & Initialization Functions ---

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
    
    for package, pip_name in required_packages.items():
        try:
            # Simplified import check
            if package == "google.generativeai":
                importlib.import_module("google.generativeai")
            elif package == "dotenv":
                importlib.import_module("dotenv")
            elif package == "dateutil":
                importlib.import_module("dateutil")
            else:
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(pip_name)
    
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

# Check requirements before proceeding (Note: You had this repeated, kept the original structure)
if not check_and_install_requirements():
    print("❌ Failed to install required packages. Exiting...")
    sys.exit(1)

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
        
        try:
            for k in (ENV_GEMINI_KEYS or []):
                self._add_key(k)
            # Placeholder for config manager access if defined later
            if 'config_manager' in globals() and hasattr(config_manager, 'config'):
                extra = config_manager.config.get('gemini_api_keys') or []
                for k in extra:
                    self._add_key(k)
        except Exception:
            pass
            
        # Ensure we don't exceed the max cap
        self.keys = self.keys[:MAX_GEMINI_KEYS]
        
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
        now = time.time()
        
        # Check starting from current index, wrap around
        for _ in range(len(self.keys)):
            k = self.keys[self.index]
            if now >= self.cooldowns.get(k, 0):
                palm.configure(api_key=k)
                return k
            self.index = (self.index + 1) % len(self.keys)
            
        # If all keys are on cooldown
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
    
# Gemini AI Configuration (Fallback if key_manager failed)
if not key_manager or not key_manager.keys:
    if GEMINI_API_KEY:
        palm.configure(api_key=GEMINI_API_KEY)
    else:
        print_status("❌ CRITICAL: No Gemini API Key found or initialized.", 'error')
model = palm.GenerativeModel("gemini-2.5-flash")

# Language Detection Function
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# GET_RANDOM_EMOJIS FUNCTION (Kept for completeness but not used in the new persona)
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
    
    # 1. Lazy initialization/reconfiguration of key manager if needed
    global key_manager
    if 'key_manager' not in globals() or key_manager is None or not key_manager.keys:
        # Re-initialize key manager if needed (simplified check)
        key_manager = GeminiKeyManager(GEMINI_API_KEY)

    try:
        # 2. Check rate limit
        if not ai_rate_limiter.can_make_request():
            return "Rate limit exceeded. Please try again later."

        # 3. Persona and Instructions
        lang_instructions = {
            # Force English and no emoji instruction for your specific persona request
            'en': 'Reply only in English language with 1-2 friendly sentences. Do not use any emojis in the response.'
        }
        
        # Set detected_lang to 'en' to use the English-only instruction
        detected_lang = 'en'
        
        templates = {
            'general': ['Keep the response natural and conversational.', 'Add some personality to the response.', 'Make the response engaging but concise.', 'Be friendly and approachable.'],
            'question': ['Provide a helpful and clear answer.', 'Be informative but keep it simple.', 'Answer directly with a friendly tone.', 'Give practical and actionable advice.'],
            'help': ['Offer assistance in a supportive way.', 'Be encouraging and helpful.', 'Provide guidance with a positive tone.', 'Show empathy and understanding.'],
            'reply': ['Acknowledge the previous message naturally.', 'Respond in a contextually appropriate way.', 'Keep the conversation flowing smoothly.', 'Build on the previous message naturally.'],
            'casual': ['Keep it casual and friendly like talking to a friend.', 'Use simple, everyday language.', 'Be relaxed and informal.'],
            'professional': ['Keep it professional but warm.', 'Be helpful and informative.', 'Maintain a helpful tone.'],
            'funny': ['Make the response humorous and entertaining.', 'Add some jokes or witty remarks.', 'Keep it light and fun.']
        }
        
        try:
            current_pattern = smart_timer.get_current_pattern()
        except Exception:
            current_pattern = 'general'
            
        if current_pattern == 'quiet':
            template = "Keep response very short and simple."
        elif current_pattern == 'casual':
            template = random.choice(templates.get('casual', templates['general']))
        elif current_pattern == 'professional':
            template = random.choice(templates.get('professional', templates['general']))
        elif current_pattern == 'funny':
            template = random.choice(templates.get('funny', templates['general']))
        else:
            template = random.choice(templates.get(message_type, templates['general']))
            
        human_variations = [
            "Make it sound like a real person typing, not a bot.",
            "Use casual, everyday language like a friend would.",
            "Sometimes add small mistakes or casual language.",
            "Keep it natural and conversational.",
            "Vary your writing style naturally.",
            "Sometimes be more formal, sometimes more casual."
        ]
        human_instruction = random.choice(human_variations)
        
        # 🇮🇳 CONSTRUCT THE FINAL, OVERRIDING PROMPT HERE
        base_identity = "You are a friendly, enthusiastic, and helpful human community member on Discord. You live in a major city in Maharashtra, India. Do not reveal that you are an AI model."
        
        full_prompt = f"{base_identity}\n" \
                      f"{lang_instructions.get('en')}\n" \
                      f"{template}\n{human_instruction}\n\n{prompt}"
                      
        # 4. Generate Content (with retry/rotation logic)
        current_key = key_manager.current_key() if key_manager else GEMINI_API_KEY
        response = None
        
        for attempt in range(2):
            try:
                if key_manager and attempt > 0:
                    current_key = key_manager.next_key()
                    if not current_key:
                        raise Exception("All Gemini keys are rate limited or invalid.")
                        
                response = model.generate_content(full_prompt)
                response_text = response.text.strip()
                break # Success
            except Exception as e:
                err_text = str(e).lower()
                if any(t in err_text for t in ["rate limit", "quota", "permission", "unauthorized", "invalid api"]):
                    # Mark key as on cooldown and try next one
                    retry_seconds = 30 
                    if key_manager and current_key:
                        key_manager.set_cooldown(current_key, retry_seconds)
                    if attempt == 0:
                        print_status(f"⚠️ Gemini API key failed ({current_key[:10]}...): {err_text}. Attempting rotation.", 'warning')
                        continue # Retry with next key
                else:
                    raise # Re-raise if not a key or rate limit issue
        
        if not response:
             return "AI Error: Failed to generate response after retries."
             
        # 5. Post-Processing
        response_text = response.text.strip()
        
        # Remove greeting openings 
        lowered = response_text.lower().lstrip()
        for greet in ["hey", "hi", "hello", "hey there", "hi there", "hello there"]:
            if lowered.startswith(greet):
                parts = response_text.split(' ', 1)
                if len(parts) == 2:
                    response_text = parts[1].lstrip("-,.!:; ")
                else:
                    response_text = response_text.lstrip("-,.!:; ")
                break
        
        # Control response length
        if len(response_text) > 200:
            response_text = response_text[:200] + "..."
            
        # Add human-like variations
        if random.random() < 0.2:  # 20% chance
            response_text = response_text.replace('.', '...').replace('!', '!!')
            
        return response_text
        
    except Exception as e:
        return f"AI Error: {e}"

# Custom Timer Function with Human-like Behavior (MODIFIED FOR BETTER ERROR CHECKING)
async def send_reply(channel_id, message, delay, message_id=None):
    """Sends a reply message to Discord with human-like typing and delay."""
    
    # 1. Simulate human typing
    typing_time = min(len(message) * 0.1, 3)  # Max 3 seconds typing
    await asyncio.sleep(typing_time)
    
    # 2. Add random delay before sending
    await asyncio.sleep(delay)
    
    data = {
        "content": message,
        # Ensure message_reference is only added if message_id exists and is not None
        "message_reference": {
            "message_id": message_id,
            "channel_id": channel_id
        } if message_id else None
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://discord.com/api/v9/channels/{channel_id}/messages", 
                headers=HEADERS, 
                json=data,
                timeout=10 # Add timeout for robustness
            )
            
            if response.status_code == 200:
                print_status(f"✅ Discord Message Sent to Channel {channel_id}", 'success')
                bot_dashboard.update_stats(response_sent=True)
                return True
            
            elif response.status_code == 429:
                # Discord Rate Limit Detected
                retry_after = int(response.headers.get('Retry-After', 5))
                print_status(f"❌ Discord Rate Limited (429). Waiting {retry_after}s.", 'error')
                await asyncio.sleep(retry_after)
                continue # Retry after sleep
                
            elif response.status_code in (401, 403, 400):
                # Unauthorized (token invalid), Forbidden (no permissions), Bad Request (bad message format)
                print_status(f"❌ Discord API Error {response.status_code}: Cannot send message (Permissions/Token/Format). {response.text[:100]}", 'error')
                return False # Fatal error, stop trying
            
            else:
                # Other HTTP errors
                print_status(f"❌ Discord HTTP Error {response.status_code}: Retrying...", 'warning')
                await asyncio.sleep(2 ** attempt) # Exponential backoff
        
        except requests.exceptions.RequestException as e:
            print_status(f"❌ Network Error during send_reply (Attempt {attempt+1}): {e}", 'error')
            await asyncio.sleep(2 ** attempt)
            
    return False


# Get Server Channels Function
def get_servers_and_channels():
    # ... (No changes needed here) ...
    try:
        guilds_response = requests.get("https://discord.com/api/v9/users/@me/guilds", headers=HEADERS)
        guilds = guilds_response.json()
        
        servers_with_channels = []
        
        for guild in guilds:
            guild_id = guild["id"]
            guild_name = guild["name"]
            
            channels_response = requests.get(f"https://discord.com/api/v9/guilds/{guild_id}/channels", headers=HEADERS)
            channels = channels_response.json()
            
            text_channels = []
            for channel in channels:
                if channel["type"] == 0:  # Type 0 is text channel
                    text_channels.append({
                        "id": channel["id"],
                        "name": channel["name"]
                    })
            
            if text_channels:
                servers_with_channels.append({
                    "id": guild_id,
                    "name": guild_name,
                    "channels": text_channels
                })
        
        return servers_with_channels
    except Exception as e:
        print(f"⚠ Error getting servers: {e}")
        return []

# Smart Break Timer Function
class SmartBreakTimer:
    # ... (No changes needed here) ...
    def __init__(self):
        self.start_time = time.time()
        self.last_break = time.time()
        self.daily_start = time.time()
        self.daily_hours = 0
        self.current_channel = None
        self.channels_used = []
        self.last_reply_time = 0  # Track when last reply was sent
        self.waiting_for_response = False  # Track if waiting for response
        self.last_reply_to_user = None  # Track who we replied to
        
        # Conversation tracking system
        self.active_conversations = {}  # Track ongoing conversations
        self.conversation_timeout = 300  # 5 minutes timeout for conversations
        self.max_conversations = 10  # keep at most 10 active users
        
        # Alternating priority flag (robust default)
        self.prefer_new_first = False
        
    def should_take_break(self):
        current_time = time.time()
        continuous_hours = (current_time - self.last_break) / 3600
        daily_hours = (current_time - self.daily_start) / 3600
        
        if continuous_hours >= ANTI_BAN_CONFIG['max_continuous_hours']:
            return True, "continuous_limit"
            
        if daily_hours >= ANTI_BAN_CONFIG['max_daily_hours']:
            return True, "daily_limit"
            
        return False, None
    
    def get_break_duration(self):
        return ANTI_BAN_CONFIG['break_duration_minutes'] * 60
        
    def should_rotate_channel(self):
        return ANTI_BAN_CONFIG['channel_rotation'] and len(self.channels_used) > 1
        
    def get_current_pattern(self):
        pattern_index = int((time.time() - self.start_time) / 3600) % len(ANTI_BAN_CONFIG['response_patterns'])
        return ANTI_BAN_CONFIG['response_patterns'][pattern_index]
        
    def add_channel(self, channel_id):
        if channel_id not in self.channels_used:
            self.channels_used.append(channel_id)
            
    def get_next_channel(self):
        if len(self.channels_used) > 1:
            current_index = self.channels_used.index(self.current_channel) if self.current_channel else 0
            next_index = (current_index + 1) % len(self.channels_used)
            return self.channels_used[next_index]
        return None
        
    def can_send_new_reply(self):
        current_time = time.time()
        
        if self.waiting_for_response:
            if current_time - self.last_reply_time > 300:  # 5 minutes
                self.waiting_for_response = False
                return True
            return False
            
        return True
        
    def mark_reply_sent(self, user_id):
        self.last_reply_time = time.time()
        self.waiting_for_response = True
        self.last_reply_to_user = user_id
        
    def check_response_received(self, messages):
        if not self.waiting_for_response or not self.last_reply_to_user:
            return False
            
        current_time = time.time()
        
        for msg in messages[:10]:
            author_id = msg.get('author', {}).get('id')
            if author_id == self.last_reply_to_user:
                timestamp = msg.get('timestamp', '')
                if timestamp:
                    try:
                        message_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        message_timestamp = message_time.timestamp()
                        
                        if current_time - message_timestamp < 120:  # 2 minutes
                            self.waiting_for_response = False
                            return True
                    except Exception:
                        pass
                        
        return False
        
    def start_conversation(self, user_id, username):
        current_time = time.time()
        self.active_conversations[user_id] = {
            'start_time': current_time,
            'last_message_time': current_time,
            'message_count': 1,
            'username': username
        }
        self.enforce_capacity()
        print_status(f"💬 Started conversation with {username}", 'info')
        
    def continue_conversation(self, user_id, username):
        if user_id in self.active_conversations:
            current_time = time.time()
            conv = self.active_conversations[user_id]
            conv['last_message_time'] = current_time
            conv['message_count'] += 1
            self.enforce_capacity()
            print_status(f"💬 Continuing conversation with {username} (Message #{conv['message_count']})", 'info')
            return True
        return False
        
    def can_continue_conversation(self, user_id):
        if user_id not in self.active_conversations:
            return False
            
        current_time = time.time()
        conv = self.active_conversations[user_id]
        time_since_last = current_time - conv['last_message_time']
        
        if time_since_last > self.conversation_timeout:
            del self.active_conversations[user_id]
            print_status(f"⏰ Conversation with {conv['username']} expired (5 min timeout)", 'warning')
            return False
            
        return True
        
    def cleanup_expired_conversations(self):
        current_time = time.time()
        expired_users = []
        
        for user_id, conv in list(self.active_conversations.items()):
            if current_time - conv['last_message_time'] > self.conversation_timeout:
                expired_users.append(user_id)
        
        for user_id in expired_users:
            username = self.active_conversations[user_id]['username']
            del self.active_conversations[user_id]
            print_status(f"⏰ Conversation with {username} expired and removed", 'warning')
            
        self.enforce_capacity()
        
    def enforce_capacity(self):
        current_time = time.time()
        expired = [uid for uid, conv in self.active_conversations.items() if current_time - conv['last_message_time'] > self.conversation_timeout]
        for uid in expired:
            username = self.active_conversations[uid]['username']
            del self.active_conversations[uid]
            print_status(f"⏰ Conversation with {username} expired and removed", 'warning')
            
        if len(self.active_conversations) > self.max_conversations:
            sorted_items = sorted(self.active_conversations.items(), key=lambda x: x[1]['last_message_time'])
            to_remove = len(self.active_conversations) - self.max_conversations
            for i in range(to_remove):
                uid, conv = sorted_items[i]
                if uid in self.active_conversations:
                    del self.active_conversations[uid]
                    print_status(f"♻️ Removed oldest conversation ({conv['username']}) to keep top {self.max_conversations}", 'info')

    def get_conversation_status(self):
        if not self.active_conversations:
            return "No active conversations"
        
        status = []
        current_time = time.time()
        for user_id, conv in self.active_conversations.items():
            time_left = self.conversation_timeout - (current_time - conv['last_message_time'])
            if time_left > 0:
                status.append(f"{conv['username']}: {int(time_left)}s left")
        
        return ", ".join(status) if status else "No active conversations"

    def set_prefer_new_first(self, value: bool):
        try:
            setattr(self, 'prefer_new_first', bool(value))
        except Exception:
            self.prefer_new_first = bool(value)
        
    def should_prefer_new_first(self) -> bool:
        return bool(getattr(self, 'prefer_new_first', False))

# Initialize smart break timer
smart_timer = SmartBreakTimer()
if not hasattr(smart_timer, 'prefer_new_first'):
    smart_timer.prefer_new_first = False

# Advanced Error Handling & Recovery System
class ErrorHandler:
    # ... (No changes needed here) ...
    def __init__(self):
        self.error_count = 0
        self.last_error_time = 0
        self.error_types = {}
        
    def handle_error(self, error, context, operation):
        current_time = time.time()
        error_type = type(error).__name__
        
        self.error_count += 1
        self.last_error_time = current_time
        
        if error_type not in self.error_types:
            self.error_types[error_type] = 0
        self.error_types[error_type] += 1
        
        print_status(f"❌ Error in {operation}: {error_type}", 'error')
        print_status(f"Context: {context}", 'warning')
        
        bot_dashboard.update_stats(error_occurred=True)
        
        if error_type == 'ConnectionError' or 'ConnectionError' in str(error):
            return self.handle_connection_error()
        elif error_type == 'RateLimitError' or '429' in str(error):
            return self.handle_rate_limit_error()
        elif error_type == 'TimeoutError' or 'Timeout' in str(error):
            return self.handle_timeout_error()
        else:
            return self.handle_generic_error()
            
    def handle_connection_error(self):
        print_status("🔄 Connection error detected. Attempting to reconnect...", 'warning')
        time.sleep(5)
        return True
        
    def handle_rate_limit_error(self):
        print_status("⏰ Rate limit reached. Waiting 30 seconds...", 'warning')
        time.sleep(30)
        return True
        
    def handle_timeout_error(self):
        print_status("⏱️ Timeout error. Retrying with longer timeout...", 'warning')
        time.sleep(3)
        return True
        
    def handle_generic_error(self):
        print_status("⚠️ Generic error. Waiting 10 seconds before retry...", 'warning')
        time.sleep(10)
        return True
        
    def should_continue_operation(self):
        if self.error_count > 10:
            return False
        if time.time() - self.last_error_time < 60:
            return False
        return True

# Initialize error handler
error_handler = ErrorHandler()

# Enhanced Function to fetch messages from a channel
async def fetch_channel_messages(channel_id, limit=20):
    # ... (No changes needed here) ...
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(
                f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}", 
                headers=HEADERS, 
                timeout=30
            )
            
            if response.status_code == 200:
                messages = response.json()
                bot_dashboard.update_stats(message_processed=True)
                return messages
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 30))
                print_status(f"⏰ Rate limited. Waiting {retry_after} seconds...", 'warning')
                await asyncio.sleep(retry_after)
            else:
                print_status(f"❌ HTTP Error {response.status_code}: {response.text}", 'error')
                error_handler.handle_error(Exception(f"HTTP {response.status_code}"), 
                                           f"Channel {channel_id}", "fetch_messages")
                
        except requests.exceptions.Timeout:
            print_status("⏱️ Request timeout. Retrying...", 'warning')
            error_handler.handle_error(Exception("Timeout"), f"Channel {channel_id}", "fetch_messages")
        except requests.exceptions.ConnectionError:
            print_status("🔌 Connection error. Retrying...", 'warning')
            error_handler.handle_error(Exception("Connection Error"), f"Channel {channel_id}", "fetch_messages")
        except Exception as e:
            print_status(f"❌ Unexpected error: {e}", 'error')
            error_handler.handle_error(e, f"Channel {channel_id}", "fetch_messages")
            
        retry_count += 1
        if retry_count < max_retries:
            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
            
    print_status("❌ Max retries reached for fetch_messages", 'error')
    return []


# Simple event logger for UI
class EventLogger:
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.events = []
        
    def add(self, text: str):
        ts = time.strftime('%H:%M:%S')
        self.events.append(f"[{ts}] {text}")
        if len(self.events) > self.capacity:
            self.events.pop(0)
            
    def last(self):
        return list(self.events)

ui_events = EventLogger(5)

# Professional Dashboard UI System
class BotDashboard:
    # ... (No changes needed here) ...
    def __init__(self):
        self.stats = {
            'messages_processed': 0,
            'responses_sent': 0,
            'errors_occurred': 0,
            'start_time': time.time(),
            'active_conversations': 0
        }
        self.status = "🟢 Online"
        self.channel_name = ""
        self.channel_id = ""
        self.mode = "casual"
        self.slowmode = 5
        self.next_refresh_eta = 0
        
    def update_stats(self, message_processed=False, response_sent=False, error_occurred=False):
        if message_processed:
            self.stats['messages_processed'] += 1
        if response_sent:
            self.stats['responses_sent'] += 1
        if error_occurred:
            self.stats['errors_occurred'] += 1
            
    def set_context(self, channel_id: str, channel_name: str, mode: str, slowmode: int, eta: int):
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.mode = mode
        self.slowmode = slowmode
        self.next_refresh_eta = eta
        
    def get_uptime(self):
        uptime_seconds = time.time() - self.stats['start_time']
        return uptime_seconds / 3600
        
    def get_response_rate(self):
        if self.stats['messages_processed'] == 0:
            return 0.0
        return (self.stats['responses_sent'] / self.stats['messages_processed']) * 100
        
    def display_dashboard(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        uptime = self.get_uptime()
        response_rate = self.get_response_rate()
        
        header = f"{Fore.YELLOW}{Style.BRIGHT}Server/Channel: {self.channel_name or self.channel_id}  |  Mode: {self.mode.upper()}  |  Slowmode: {self.slowmode}s  |  Next Refresh: ~{self.next_refresh_eta}s{Style.RESET_ALL}"
        print("=" * max(80, len(header)))
        print(header)
        print("=" * max(80, len(header)))
        
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                               🤖 ADVANCED DISCORD BOT DASHBOARD             ║")
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        print(f"║ Status: {self.status:<65} ║")
        print(f"║ Uptime: {uptime:.1f} hours{' ' * 55} ║")
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        print(f"║ 📊 Messages Processed: {self.stats['messages_processed']:<45} ║")
        print(f"║ 💬 Responses Sent: {self.stats['responses_sent']:<48} ║")
        print(f"║ ⚠️  Errors Occurred: {self.stats['errors_occurred']:<46} ║")
        print(f"║ 📈 Response Rate: {response_rate:.1f}%{' ' * 50} ║")
        print(f"║ 💭 Active Conversations: {self.stats['active_conversations']:<42} ║")
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        print("║ Recent Events:                                                               ║")
        last = ui_events.last()
        if not last:
            print("║   (no recent events)                                                         ║")
        else:
            for ev in last[-5:]:
                line = ev[:74]
                print(f"║ {line:<74} ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")

# Enhanced Terminal UI Functions
def print_header(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{text.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")

def print_status(text, status_type='info'):
    colors = {
        'success': Fore.GREEN,
        'error': Fore.RED,
        'info': Fore.BLUE,
        'warning': Fore.YELLOW
    }
    color = colors.get(status_type, Fore.WHITE)
    print(f"{color}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def show_progress_bar(description, current, total, width=50):
    # ... (Not used in the main loop, kept as is) ...
    progress = int(width * current / total)
    bar = '█' * progress + '░' * (width - progress)
    percentage = current / total * 100
    
    sys.stdout.write(f'\r{Fore.CYAN}{description}: [{bar}] {percentage:.1f}%{Style.RESET_ALL}')
    sys.stdout.flush()
    
    if current == total:
        print()

# Initialize dashboard
bot_dashboard = BotDashboard()

# Cooldown tracking (per-user)
USER_COOLDOWNS = {}

def is_on_cooldown(user_id: str, min_seconds: int = 60, max_seconds: int = 120) -> bool:
    now = time.time()
    window = USER_COOLDOWNS.get(user_id)
    if window and now < window:
        return True
    # Not on cooldown; assign a new randomized cooldown end
    USER_COOLDOWNS[user_id] = now + random.randint(min_seconds, max_seconds)
    return False

# Config manager
class ConfigManager:
    # ... (No changes needed here) ...
    def __init__(self, path: str = "config.json"):
        self.path = path
        self.last_mtime = 0
        self.config = {
            "default_mode": "casual",
            "cooldown_seconds": {"min": 60, "max": 120},
            "channels": {"default_slowmode": 5, "overrides": {}},
            "response_patterns": ["casual", "professional", "funny", "helpful", "quiet"],
            "webhook_url": "",
            "owner_id": "",
            "gemini_api_keys": [], # Added for multiple API keys
            "ignored_usernames": ["Skbindas", "abhi$", "1050008136846671922"],  
            "ignored_user_ids": ["1050008136846671922"]
        }
        self.load(force=True)
        
    def load(self, force: bool = False):
        try:
            if not os.path.exists(self.path):
                return
            mtime = os.path.getmtime(self.path)
            if force or mtime != self.last_mtime:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.config.update(data)
                        self.config["ignored_usernames"] = data.get("ignored_usernames", self.config["ignored_usernames"])
                        self.config["ignored_user_ids"] = data.get("ignored_user_ids", self.config["ignored_user_ids"])
                        self.last_mtime = mtime
                        print_status("🔁 Config loaded", 'info')
        except Exception as e:
            print_status(f"Config load error: {e}", 'error')
            
    def get_owner_id(self):
        return self.config.get("owner_id") or os.getenv("BOT_OWNER_ID", "")
        
    def get_mode(self):
        return self.config.get("default_mode", "casual")
        
    def get_slowmode(self, channel_id: str):
        overrides = self.config.get("channels", {}).get("overrides", {})
        if channel_id in overrides and isinstance(overrides[channel_id], int):
            return overrides[channel_id]
        return self.config.get("channels", {}).get("default_slowmode", 5)
        
    def get_cooldown_range(self):
        cd = self.config.get("cooldown_seconds", {})
        return int(cd.get("min", 60)), int(cd.get("max", 120))
        
    def get_ignored_usernames(self):
        return self.config.get("ignored_usernames", [])
        
    def get_ignored_user_ids(self):
        return self.config.get("ignored_user_ids", [])


# Initialize config
config_manager = ConfigManager()

# Startup validation checks
def validate_startup():
    # ... (No changes needed here) ...
    problems = []

    if not DISCORD_TOKEN or len(DISCORD_TOKEN) < 10:
        problems.append("DISCORD_TOKEN is missing/invalid in .env")
    if not GEMINI_API_KEY and not ENV_GEMINI_KEYS:
        problems.append("GEMINI_API_KEY or GEMINI_API_KEYS is missing/invalid in .env")

    try:
        resp = requests.get("https://discord.com/api/v9/users/@me", headers=HEADERS, timeout=15)
        if resp.status_code not in (200, 401):
            problems.append(f"Discord API unusual status: {resp.status_code}")
        elif resp.status_code == 401:
            problems.append("Discord token unauthorized (401). Check DISCORD_TOKEN format.")
    except Exception as e:
        problems.append(f"Discord reachability error: {e}")

    try:
        # Check model reachability using a simple prompt
        _ = model.generate_content("ping")  
    except Exception as e:
        err = str(e).lower()
        if any(t in err for t in ["rate limit", "quota", "retry_delay", "exceeded"]):
            print_status(f"⚠ Gemini quota warning: {e}", 'warning')
        else:
            problems.append(f"Gemini check failed: {e}")

    if problems:
        print_header("Startup Checks Failed")
        for p in problems:
            print_status(f"- {p}", 'error')
        return False

    print_status("✅ Startup checks passed", 'success')
    return True

# Selfbot Main Function (FIXED: Uses get_gemini_response and removed duplicate logic)
async def selfbot():
    print_header("Discord Chat Bot")
    print_status("Bot is starting...", 'info')

    if not validate_startup():
        print_status("Fix the above issues and restart the bot.", 'error')
        return
    
    # Input and Channel Setup
    channels_input = input(f"{Fore.CYAN}👉 Enter channel IDs (separate with comma for multiple): {Style.RESET_ALL}").strip()
    if not channels_input:
        print_status("Channel IDs cannot be empty. Please try again.", 'error')
        return
    
    channel_ids = [cid.strip() for cid in channels_input.split(',')]
    valid_channels = []
    
    for cid in channel_ids:
        if not cid.isdigit():
            print_status(f"Invalid channel ID: {cid}. Skipping...", 'warning')
            continue
        valid_channels.append(cid)
    
    if not valid_channels:
        print_status("No valid channel IDs provided. Exiting...", 'error')
        return
    
    channel_id = valid_channels[0]
    for cid in valid_channels:
        smart_timer.add_channel(cid)
    
    print_status(f"✅ Added {len(valid_channels)} channels for rotation", 'success')

    if channel_id not in CHANNEL_SLOW_MODES:
        while True:
            slow_mode_input = input(f"{Fore.CYAN}🔄 Enter Slow Mode (seconds, default 5): {Style.RESET_ALL}").strip()
            if not slow_mode_input:
                slow_mode = random.randint(3, 8)
                break
            try:
                slow_mode = int(slow_mode_input)
                if slow_mode < 0:
                    print_status("Slow mode must be a positive number. Please try again.", 'error')
                    continue
                break
            except ValueError:
                print_status("Please enter a valid number for slow mode.", 'error')
        CHANNEL_SLOW_MODES[channel_id] = slow_mode
    
    smart_timer.add_channel(channel_id)
    smart_timer.current_channel = channel_id
    
    print_status("✅ Bot successfully initialized!", 'success')
    
    bot_dashboard.display_dashboard()
    
    BOT_OWNER_ID = os.getenv("BOT_OWNER_ID", "")
    CURRENT_MODE = "casual"
    
    IGNORED_USERS = config_manager.get_ignored_usernames()
    IGNORED_USER_IDS = config_manager.get_ignored_user_ids()

    while True:
        try:
            # Config Reload and Anti-Ban Checks
            config_manager.load()
            
            BOT_OWNER_ID = config_manager.get_owner_id()
            CURRENT_MODE = config_manager.get_mode()
            CHANNEL_SLOW_MODES[channel_id] = config_manager.get_slowmode(channel_id)
            cd_min, cd_max = config_manager.get_cooldown_range()
            IGNORED_USERS = config_manager.get_ignored_usernames()
            IGNORED_USER_IDS = config_manager.get_ignored_user_ids()
            
            should_break, break_reason = smart_timer.should_take_break()
            if should_break:
                ui_events.add(f"Taking break: {break_reason}")
                break_duration = smart_timer.get_break_duration()
                print_status(f"🛡️ Taking anti-ban break for {break_duration//60} minutes...", 'warning')
                
                for i in range(break_duration//60, 0, -1):
                    print(f"\r{Fore.YELLOW}⏰ Break remaining: {i} minutes{Style.RESET_ALL}", end='')
                    await asyncio.sleep(60)
                print()
                
                smart_timer.last_break = time.time()
                print_status("✅ Break completed! Resuming...", 'success')
                continue
                
            if smart_timer.should_rotate_channel():
                new_channel = smart_timer.get_next_channel()
                if new_channel and new_channel != channel_id:
                    ui_events.add(f"Rotating channel -> {new_channel}")
                    print_status(f"🔄 Rotating to channel: {new_channel}", 'info')
                    channel_id = new_channel
                    smart_timer.current_channel = channel_id
            
            # Fetch Messages
            messages = await fetch_channel_messages(channel_id, 20)
            
            if not messages:
                print("❌ No messages found or error occurred.")
                await asyncio.sleep(10)
                continue
            
            smart_timer.cleanup_expired_conversations()
            response_received = smart_timer.check_response_received(messages)
            if response_received:
                ui_events.add("User responded; resuming session")
                print_status(f"✅ Response received from user! Ready for new messages.", 'success')
            
            # Display recent messages
            print_header("Recent Messages")
            for i, msg in enumerate(messages[:20]):
                author = msg.get("author", {}).get("username", "Unknown")
                content = msg.get("content", "")
                if content:
                    truncated_content = f"{content[:50]}..." if len(content) > 50 else content
                    print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {Fore.YELLOW}{author}{Style.RESET_ALL}: {truncated_content}")
            
            # Waiting state handling
            waiting = False
            remaining_wait = 0
            has_convo_reply = False
            convo_user_id = smart_timer.last_reply_to_user
            
            if smart_timer.waiting_for_response:
                remaining_wait = max(0, int(300 - (time.time() - smart_timer.last_reply_time)))
                if remaining_wait > 0:
                    waiting = True
                    # Check for conversation reply (redundant, but helps logic flow)
                    for m in messages[:20]:
                        if m.get('author', {}).get('id') == convo_user_id:
                            ts = m.get('timestamp', '')
                            if ts:
                                mt = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                                if time.time() - mt <= 300:
                                    has_convo_reply = True
                                    break
                    
                    if not has_convo_reply:
                        print(f"\r{Fore.YELLOW}⏳ Waiting for conversation reply... ({remaining_wait}s left){Style.RESET_ALL}", end='')
                    
                else:
                    smart_timer.waiting_for_response = False

            # Message Processing Logic
            convo_msgs = [msg for msg in messages[:20] if smart_timer.can_continue_conversation(msg.get('author', {}).get('id'))]
            new_msgs = [msg for msg in messages[:20] if not smart_timer.can_continue_conversation(msg.get('author', {}).get('id'))]

            if waiting and has_convo_reply:
                # Prioritize the conversation user's latest reply
                convo_msgs = [m for m in convo_msgs if m.get('author', {}).get('id') == convo_user_id]
                new_msgs = []
            elif waiting and not has_convo_reply:
                # If waiting and they haven't replied, only consider new users
                convo_msgs = []

            process_order = [new_msgs, convo_msgs] if smart_timer.should_prefer_new_first() else [convo_msgs, new_msgs]
            
            if not waiting:
                smart_timer.set_prefer_new_first(not smart_timer.should_prefer_new_first())

            reply_sent = False
            bot_user_id = DISCORD_TOKEN.split(".")[0] if DISCORD_TOKEN and '.' in DISCORD_TOKEN else None
            
            for bucket in process_order:
                for msg in bucket:
                    author = msg.get("author", {}).get("username", "Unknown")
                    user_id = msg.get('author', {}).get('id')
                    content = msg.get("content", "")
                    
                    # Ignore own messages and ignored users
                    if user_id == bot_user_id:
                        continue
                    if user_id in IGNORED_USER_IDS or author in IGNORED_USERS:
                        print_status(f"🚫 Ignoring message from specified user: {author}", 'info')
                        continue
                        
                    # Recency Check
                    try:
                        timestamp = msg.get('timestamp', '')
                        if not timestamp: continue
                        message_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        message_timestamp = message_time.timestamp()
                        if time.time() - message_timestamp > 120: continue # Max 2 minute old messages
                    except Exception:
                        continue
                        
                    # Decision Logic
                    should_respond = False
                    message_type = 'general'
                    context = None
                    content_lower = content.lower()
                    
                    if smart_timer.can_continue_conversation(user_id):
                        should_respond = True
                        message_type = 'conversation'
                        smart_timer.continue_conversation(user_id, author)
                        context = f"This is a continuation of a conversation. {author} says: {content}"
                    
                    elif msg.get("referenced_message") is not None:
                        should_respond = True
                        message_type = 'reply'
                        context = f"Reply to a previous message. {author} replied: {content}"
                        
                    elif ('?' in content_lower and len(content_lower) > 10) or any(q in content_lower for q in ['what is','how do','why does','when will','where can','who is','which one']):
                        should_respond = True
                        message_type = 'question'
                        context = f"Someone asked a question. {author} asked: {content}"
                        
                    elif any(word in content_lower for word in ['help','how','what','why','when','where','who','please','could','would']):
                        should_respond = True
                        message_type = 'help'
                        context = f"Someone needs assistance. {author} said: {content}"
                        
                    if not should_respond:
                        continue
                        
                    # Cooldown Check (Only apply after a decision to respond is made)
                    if is_on_cooldown(user_id, cd_min, cd_max):
                        continue
                        
                    # Random Skip Chance (20%)
                    if random.random() < 0.2:
                        continue
                        
                    # Generate and Send Reply (FIXED SECTION)
                    if context:
                        # 1. Get the response using the dedicated function
                        detected_lang = detect_language(content)
                        # We use a base prompt that get_gemini_response will wrap with the persona and constraints
                        prompt_for_ai = f"Reply to this message naturally and appropriately. Message from {author}: {content}"
                        
                        # The function handles all AI key rotation and rate limit retries internally
                        ai_response = get_gemini_response(prompt_for_ai, detected_lang, message_type)
                        
                        if ai_response and not ai_response.startswith(("AI Error", "Rate limit")):
                            # 2. Send the reply using the human-like delay function
                            ui_events.add(f"Reply -> {author}")
                            human_delay = random.randint(2, 8)
                            
                            # send_reply will handle retries and report errors
                            send_success = await send_reply(channel_id, ai_response, human_delay, msg.get('id'))
                            
                            if send_success:
                                # 3. Update conversation state only if send was successful
                                smart_timer.mark_reply_sent(user_id)
                                if message_type != 'conversation':
                                    smart_timer.start_conversation(user_id, author)
                                    
                                reply_sent = True
                                print_status(f"⏳ Waiting for response from {author} (5 minutes timeout)", 'info')
                                break # Stop processing messages in this bucket/loop
                            else:
                                # Send failed (e.g., Discord rate limit or permission error)
                                ui_events.add(f"Failed to send reply to {author}")
                                # Do not mark reply_sent, allow next bucket/channel in next cycle
                                
                        else:
                            # AI generation failed (internal API error or rate limit)
                            ui_events.add(f"AI Error/Limit: {ai_response}")
                            await asyncio.sleep(5) # Wait a bit before next attempt/channel switch
                            
                if reply_sent:
                    break # Break out of bucket processing if a reply was sent successfully

            # Update dashboard and wait
            bot_dashboard.stats['active_conversations'] = len(smart_timer.active_conversations)
            bot_dashboard.set_context(channel_id, channel_id, CURRENT_MODE, CHANNEL_SLOW_MODES.get(channel_id, 5), CHANNEL_SLOW_MODES.get(channel_id, 5))
            bot_dashboard.display_dashboard()
            
            # Wait for slow mode duration with progress indicator
            slow_mode_time = CHANNEL_SLOW_MODES.get(channel_id, 5)
            for i in range(slow_mode_time, 0, -1):
                print(f"\r{Fore.CYAN}⏳ Refreshing in {i} seconds...{Style.RESET_ALL}", end='')
                await asyncio.sleep(1)
            print() # New line after countdown
            
        except Exception as e:
            print(f"⚠ Error in main loop: {e}")
            error_handler.handle_error(e, "Main Loop", "selfbot")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(selfbot())
