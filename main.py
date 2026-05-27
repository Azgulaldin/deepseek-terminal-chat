# =========================================================
# DeepSeek Persistent Thinking Chat
# =========================================================

import json
import os
import re
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


# =========================================================
# FILE PATHS
# =========================================================

BASE_FOLDER = Path(r"C:\Boobies")

PREFERENCES_FILE = BASE_FOLDER / "user prefrances.txt"
CHAT_FOLDER = BASE_FOLDER / "chat list"

CHAT_FOLDER.mkdir(parents=True, exist_ok=True)


# =========================================================
# DEFAULT PREFERENCES
# =========================================================

DEFAULT_PREFERENCES: Dict[str, Any] = {
    "api_key": "boobies of death",
    "user_name": "Ad",
    "assistant_behavior": "Be helpful, clear, and natural.",
    "reasoning_enabled": True,
    "reasoning_effort": "max",
    "show_reasoning": False,
    "generate_titles": True,
}


# =========================================================
# PREFERENCES
# =========================================================

def normalize_preferences(prefs: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_PREFERENCES)
    merged.update(prefs or {})

    merged["api_key"] = str(merged.get("api_key", "")).strip()

    merged["user_name"] = (
        str(merged.get("user_name", "")).strip()
        or DEFAULT_PREFERENCES["user_name"]
    )

    merged["assistant_behavior"] = (
        str(merged.get("assistant_behavior", "")).strip()
        or DEFAULT_PREFERENCES["assistant_behavior"]
    )

    merged["reasoning_enabled"] = bool(
        merged.get(
            "reasoning_enabled",
            DEFAULT_PREFERENCES["reasoning_enabled"]
        )
    )

    merged["show_reasoning"] = bool(
        merged.get(
            "show_reasoning",
            DEFAULT_PREFERENCES["show_reasoning"]
        )
    )

    merged["generate_titles"] = bool(
        merged.get(
            "generate_titles",
            DEFAULT_PREFERENCES["generate_titles"]
        )
    )

    effort = str(
        merged.get(
            "reasoning_effort",
            DEFAULT_PREFERENCES["reasoning_effort"]
        )
    ).lower().strip()

    if effort not in {"low", "medium", "high", "max"}:
        effort = DEFAULT_PREFERENCES["reasoning_effort"]

    merged["reasoning_effort"] = effort

    return merged


def save_preferences(prefs: Dict[str, Any]) -> None:
    with PREFERENCES_FILE.open("w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)


def load_preferences() -> Dict[str, Any]:
    if not PREFERENCES_FILE.exists():
        prefs = normalize_preferences(DEFAULT_PREFERENCES)
        save_preferences(prefs)
        return prefs

    try:
        with PREFERENCES_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return normalize_preferences(data)

    except Exception as e:
        print(f"Failed loading preferences: {e}")

        prefs = normalize_preferences(DEFAULT_PREFERENCES)
        save_preferences(prefs)

        return prefs


# =========================================================
# INPUT HELPERS
# =========================================================

def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"

    while True:
        value = input(f"{prompt} {suffix}: ").strip().lower()

        if not value:
            return default

        if value in {"y", "yes"}:
            return True

        if value in {"n", "no"}:
            return False

        print("Enter y or n.")


def ask_text(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def ask_secret(prompt: str, default: str = "") -> str:
    hint = "[set]" if default else "[empty]"

    while True:
        raw = getpass(f"{prompt} {hint}: ").strip()

        if raw:
            return raw

        if default:
            return default

        print("Cannot be blank.")


def ask_reasoning_effort(default: str = "max") -> str:
    allowed = {"low", "medium", "high", "max"}

    while True:
        value = ask_text(
            "Reasoning strength (low / medium / high / max)",
            default
        ).lower()

        if value in allowed:
            return value

        print("Invalid option.")


# =========================================================
# PREFERENCE EDITOR
# =========================================================

def edit_preferences_interactively(current: Dict[str, Any]) -> Dict[str, Any]:

    print("\nEdit Preferences\n")

    updated = dict(current)

    updated["api_key"] = ask_secret(
        "DeepSeek API key",
        current.get("api_key", "")
    )

    updated["user_name"] = ask_text(
        "What should the user be called",
        current["user_name"]
    )

    updated["assistant_behavior"] = ask_text(
        "How should the AI behave",
        current["assistant_behavior"]
    )

    updated["reasoning_enabled"] = ask_yes_no(
        "Enable reasoning",
        current["reasoning_enabled"]
    )

    if updated["reasoning_enabled"]:
        updated["reasoning_effort"] = ask_reasoning_effort(
            current["reasoning_effort"]
        )

    updated["show_reasoning"] = ask_yes_no(
        "Show reasoning output",
        current["show_reasoning"]
    )

    updated["generate_titles"] = ask_yes_no(
        "Generate titles",
        current["generate_titles"]
    )

    updated = normalize_preferences(updated)

    save_preferences(updated)

    print("\nPreferences saved.\n")

    return updated


# =========================================================
# API
# =========================================================

def resolve_api_key(prefs: Dict[str, Any]) -> str:

    stored = prefs.get("api_key", "").strip()

    if stored:
        return stored

    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if env_key:
        return env_key

    print("\nNo API key found.\n")

    key = ask_secret("Enter DeepSeek API key")

    prefs["api_key"] = key
    save_preferences(prefs)

    return key


def create_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )


# =========================================================
# CHAT SESSION STORAGE
# =========================================================

def sanitize_filename(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_existing_chat_files():
    files = []

    for file in CHAT_FOLDER.glob("*.json"):
        match = re.match(r"^(\d+)_", file.name)

        if match:
            number = int(match.group(1))
            files.append((number, file))

    files.sort(key=lambda x: x[0])

    return files


def print_chat_list():

    files = get_existing_chat_files()

    print("\n=========== CHAT LIST ===========")

    if not files:
        print("No previous chats.")

    for number, file in files:
        print(f"{number}. {file.stem}")

    print("=================================\n")


def get_next_chat_number() -> int:

    files = get_existing_chat_files()

    if not files:
        return 1

    return max(num for num, _ in files) + 1


def create_new_chat_file() -> Path:

    number = get_next_chat_number()

    date = datetime.now().strftime("%Y-%m-%d_%H-%M")

    filename = f"{number}_New Chat_{date}.json"

    return CHAT_FOLDER / filename


def save_chat_file(chat_path: Path, data: Dict[str, Any]) -> None:

    with chat_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chat_file(chat_path: Path) -> Dict[str, Any]:

    with chat_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rename_chat_file(chat_path: Path, title: str) -> Path:

    number_match = re.match(r"^(\d+)_", chat_path.name)

    if not number_match:
        return chat_path

    number = number_match.group(1)

    date = datetime.now().strftime("%Y-%m-%d_%H-%M")

    safe_title = sanitize_filename(title)

    new_name = f"{number}_{safe_title}_{date}.json"

    new_path = CHAT_FOLDER / new_name

    try:
        if chat_path.exists():
            chat_path.rename(new_path)
            return new_path
    except:
        pass

    return chat_path


# =========================================================
# TITLES
# =========================================================

def fallback_title(text: str) -> str:

    words = text.strip().split()

    if not words:
        return "Untitled"

    return " ".join(words[:6])


def generate_title(client: OpenAI, text: str) -> str:

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a concise title between "
                    "3 and 6 words."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    title = response.choices[0].message.content or ""

    return title.strip() or fallback_title(text)


# =========================================================
# MEMORY IMPORT
# =========================================================

def load_memory_into_system_prompt(selected_numbers):

    memory_parts = []

    files = dict(get_existing_chat_files())

    for number in selected_numbers:

        if number not in files:
            continue

        try:
            data = load_chat_file(files[number])

            history = data.get("conversation_history", [])

            memory_parts.append(
                f"\n=== MEMORY FROM CHAT {number} ===\n"
            )

            for msg in history:

                role = msg.get("role", "").upper()
                content = msg.get("content", "")

                memory_parts.append(
                    f"{role}: {content}\n"
                )

        except:
            continue

    return "\n".join(memory_parts)


# =========================================================
# MAIN
# =========================================================

def main():

    print("DeepSeek Persistent Thinking Chat")
    print("Type 'quit' to exit.")
    print("Type 'prefs' to edit preferences.\n")

    prefs = load_preferences()

    edit_now = ask_yes_no(
        "Edit preferences now",
        False
    )

    if edit_now:
        prefs = edit_preferences_interactively(prefs)

    api_key = resolve_api_key(prefs)

    client = create_client(api_key)

    # =====================================================
    # CHAT SELECTION
    # =====================================================

    print_chat_list()

    choice = input(
        "Enter chat number to continue "
        "or type 'new': "
    ).strip().lower()

    conversation_history = []

    chat_path = None

    if choice == "new":

        chat_path = create_new_chat_file()

        memory_text = ""

        use_memory = ask_yes_no(
            "Load previous chats as memory",
            False
        )

        if use_memory:

            print_chat_list()

            raw = input(
                "Enter chat numbers separated "
                "by commas: "
            ).strip()

            selected = []

            for part in raw.split(","):

                part = part.strip()

                if part.isdigit():
                    selected.append(int(part))

            memory_text = load_memory_into_system_prompt(
                selected
            )

        system_prompt = (
            "You are a highly intelligent and "
            "helpful assistant.\n"
            f"Address the user as "
            f"{prefs['user_name']}.\n"
            f"Behavior guidance: "
            f"{prefs['assistant_behavior']}\n"
        )

        if memory_text:
            system_prompt += (
                "\nUse the following as long-term "
                "memory and context:\n"
                + memory_text
            )

        conversation_history = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        save_chat_file(chat_path, {
            "conversation_history": conversation_history
        })

    else:

        selected_number = int(choice)

        files = dict(get_existing_chat_files())

        if selected_number not in files:
            print("Chat not found.")
            return

        chat_path = files[selected_number]

        data = load_chat_file(chat_path)

        conversation_history = data.get(
            "conversation_history",
            []
        )

    # =====================================================
    # SESSION START
    # =====================================================

    print("\nSession ready.\n")

    current_title = "New Chat"

    while True:

        user_input = input(
            f"{prefs['user_name']}: "
        ).strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Session ended.")
            break

        if user_input.lower() == "prefs":

            prefs = edit_preferences_interactively(
                prefs
            )

            api_key = resolve_api_key(prefs)

            client = create_client(api_key)

            continue

        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        save_chat_file(chat_path, {
            "conversation_history": conversation_history
        })

        request_kwargs = {
            "model": "deepseek-v4-pro",
            "messages": conversation_history,
        }

        if prefs["reasoning_enabled"]:

            request_kwargs["reasoning_effort"] = (
                prefs["reasoning_effort"]
            )

            request_kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled"
                }
            }

        try:

            response = client.chat.completions.create(
                **request_kwargs
            )

        except Exception as e:

            print(f"\nError: {e}\n")
            continue

        message = response.choices[0].message

        final_answer = (
            message.content or ""
        ).strip()

        reasoning = getattr(
            message,
            "reasoning_content",
            None
        )

        conversation_history.append({
            "role": "assistant",
            "content": final_answer
        })

        save_chat_file(chat_path, {
            "conversation_history": conversation_history
        })

        # =================================================
        # AUTO TITLE + RENAME
        # =================================================

        if (
            prefs["generate_titles"]
            and current_title == "New Chat"
        ):

            try:
                current_title = generate_title(
                    client,
                    final_answer
                )

            except:
                current_title = fallback_title(
                    final_answer
                )

            chat_path = rename_chat_file(
                chat_path,
                current_title
            )

        print("\n======================================")

        print(f"TITLE: {current_title}")

        print("======================================")

        if (
            prefs["show_reasoning"]
            and prefs["reasoning_enabled"]
            and reasoning
        ):

            print("\n=========== REASONING ===========\n")
            print(reasoning)

        print("\n========= FINAL ANSWER =========\n")

        print(final_answer)

        print("\n======================================\n")


if __name__ == "__main__":
    main()


# =========================================================
# END
# =========================================================