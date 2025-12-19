#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Tkinter-based front-end for the BI-agent.

This application provides a two‑column interface:

* On the left: a history of past chats loaded from MinIO. Users can click
  on a chat to view all its recorded steps (initial prompt, clarifications,
  analytics, and visualization).
* On the right: the conversation view for the selected chat or an active
  session. A new chat can be started, allowing the user to send a request
  and interact with the agent. As the agent progresses through its
  workflow, the UI displays each step in real time.

MinIO connection parameters are read from the environment variables
`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` and
`MINIO_BUCKET_NAME`. If these variables are unset, sensible defaults are
used (matching those in the agent code). The application assumes that
objects in MinIO are stored following the logging scheme implemented in
`test.py`: JSON files under `<chat_id>/<step_num>_<type>.json` and, if
applicable, image files under `<chat_id>/<step_num>_viz.png`.

Limitations:

* This front‑end does not perform error handling for missing/invalid
  MinIO configuration. If MinIO is unreachable, the history will not be
  populated.
* Interaction with the underlying agent is provided as a skeleton. The
  agent functions should be imported from `test.py` (or another module)
  and integrated into `send_message()` according to your environment.
* Image display relies on Pillow (`PIL`). If Pillow is not installed,
  images are skipped.
"""

import os
import json
import uuid
import io
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional  # For backward compatibility with Python < 3.10

# Try to import PIL for image handling. If unavailable, images will be skipped.
try:
    from PIL import Image, ImageTk  # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Attempt to import the MinIO client. The application requires `minio`
# to be installed in order to retrieve and display chat history.
try:
    from minio import Minio
    from minio.error import S3Error  # noqa: F401
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

# Optional: import the agent functions. Adjust the import path as needed.
try:
    import test  # type: ignore
    AGENT_AVAILABLE = True
except Exception:
    AGENT_AVAILABLE = False


class ChatApp(tk.Tk):
    """Main application class for the BI-agent front‑end."""

    def __init__(self):
        super().__init__()
        self.title("BI-agent Frontend")
        self.geometry("1000x600")
        self.configure(bg="white")

        # Data structures
        self.chats = {}  # chat_id -> chat_name
        self.current_chat_id: Optional[str] = None
        self.current_state: Optional[dict] = None

        # Initialize MinIO client
        self.minio_client = None
        if MINIO_AVAILABLE:
            endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
            # Remove scheme if present
            endpoint = endpoint.replace("https://", "").replace("http://", "")
            access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
            secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
            bucket_name = os.getenv("MINIO_BUCKET_NAME", "bi-agent-logs")
            self.bucket_name = bucket_name
            # Determine secure parameter based on MINIO_SECURE or presence of https in endpoint
            secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes")
            try:
                self.minio_client = Minio(
                    endpoint=endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=secure,
                )
            except Exception as e:
                messagebox.showerror("MinIO error", f"Could not connect to MinIO: {e}")
                self.minio_client = None
        else:
            self.bucket_name = None
            messagebox.showwarning(
                "MinIO unavailable",
                "The 'minio' package is not installed. Chat history cannot be loaded."
            )

        # Build UI components
        self._create_widgets()

        # Load chat history
        if self.minio_client is not None:
            threading.Thread(target=self.load_history, daemon=True).start()

    def _create_widgets(self):
        """Set up the main UI layout."""
        # Left panel for chat history
        left_frame = tk.Frame(self, width=200, bg="#F5F5F5", relief=tk.SUNKEN, bd=1)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Header and new chat button
        header = tk.Label(left_frame, text="История запросов", bg="#F5F5F5", font=("Arial", 12, "bold"))
        header.pack(pady=(10, 5))

        new_chat_btn = tk.Button(left_frame, text="Новый чат", command=self.start_new_chat)
        new_chat_btn.pack(pady=(0, 10))

        # Listbox for chat names
        self.chat_listbox = tk.Listbox(left_frame)
        self.chat_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        self.chat_listbox.bind("<<ListboxSelect>>", self._on_chat_select)

        # Right panel for conversation
        right_frame = tk.Frame(self, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas with scrollbar for conversation history
        self.canvas = tk.Canvas(right_frame, bg="white")
        self.scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Frame inside the canvas to hold messages
        self.conv_frame = tk.Frame(self.canvas, bg="white")
        self.canvas.create_window((0, 0), window=self.conv_frame, anchor="nw")
        self.conv_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Entry and send button at bottom
        input_frame = tk.Frame(self, bg="white")
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.send_btn = tk.Button(input_frame, text="Отправить", command=self.send_message)
        self.send_btn.pack(side=tk.RIGHT)
        self.send_btn.config(state=tk.DISABLED)  # disabled until chat started

    def load_history(self):
        """Fetch chat history from MinIO and populate the listbox."""
        if self.minio_client is None or self.bucket_name is None:
            return
        try:
            # Use a set to track seen chat IDs
            seen = {}
            # List all JSON objects in the bucket
            objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
            for obj in objects:
                if not obj.object_name.endswith(".json"):
                    continue
                try:
                    response = self.minio_client.get_object(self.bucket_name, obj.object_name)
                    data = response.read().decode("utf-8")
                    record = json.loads(data)
                except Exception:
                    continue
                chat_id = record.get("chat_id")
                chat_name = record.get("chat_name", chat_id)
                if chat_id and chat_id not in seen:
                    seen[chat_id] = chat_name
            # sort by chat_name or creation order; we'll sort by chat_name for now
            self.chats = seen
            # Update UI on main thread
            self.after(0, self._refresh_history_list)
        except Exception as e:
            messagebox.showerror("History error", f"Failed to load history: {e}")

    def _refresh_history_list(self):
        """Populate the listbox with chat names."""
        self.chat_listbox.delete(0, tk.END)
        for chat_id, chat_name in self.chats.items():
            self.chat_listbox.insert(tk.END, chat_name)

    def _on_chat_select(self, event):
        """Handler for selecting a chat from the history."""
        if not self.chat_listbox.curselection():
            return
        index = self.chat_listbox.curselection()[0]
        chat_name = self.chat_listbox.get(index)
        # Find chat_id by name (assumes unique names)
        chat_id = None
        for cid, name in self.chats.items():
            if name == chat_name:
                chat_id = cid
                break
        if chat_id:
            self.display_chat(chat_id)

    def display_chat(self, chat_id: str):
        """Load and display all messages for the given chat ID."""
        # Clear current content
        for widget in self.conv_frame.winfo_children():
            widget.destroy()
        # Disable send button since we are in history mode
        self.send_btn.config(state=tk.DISABLED)
        self.input_entry.delete(0, tk.END)
        self.current_chat_id = None
        self.current_state = None

        if self.minio_client is None or self.bucket_name is None:
            return
        try:
            objects = self.minio_client.list_objects(self.bucket_name, prefix=f"{chat_id}/", recursive=True)
            steps = []
            for obj in objects:
                if obj.object_name.endswith(".json"):
                    try:
                        response = self.minio_client.get_object(self.bucket_name, obj.object_name)
                        data = response.read().decode("utf-8")
                        record = json.loads(data)
                        steps.append(record)
                    except Exception:
                        continue
            # Sort by step number
            steps.sort(key=lambda x: x.get("step_num", 0))
            # Display each step
            for step in steps:
                typ = step.get("type_of_content")
                content = step.get("content")
                if typ == "image_link":
                    # Extract object name from chat_id and step number
                    step_num = step.get("step_num")
                    image_obj = f"{chat_id}/{step_num}_viz.png"
                    image_data = None
                    if PIL_AVAILABLE and self.minio_client is not None:
                        try:
                            resp = self.minio_client.get_object(self.bucket_name, image_obj)
                            image_bytes = resp.read()
                            image_data = Image.open(io.BytesIO(image_bytes))
                        except Exception:
                            image_data = None
                    self.append_message("\n[Визуализация]", image=image_data)
                else:
                    label = self._type_to_label(typ)
                    self.append_message(f"{label}:\n{content}")
        except Exception as e:
            messagebox.showerror("Load chat error", f"Unable to load chat: {e}")

    def _type_to_label(self, typ: Optional[str]) -> str:
        # Use Optional[str] for Python versions earlier than 3.10
        mapping = {
            "initial_prompt": "Запрос",
            "clarification_questions": "Уточнение",
            "clarification_response": "Ответ",
            "analytics": "Аналитика",
            "image_link": "Визуализация",
        }
        return mapping.get(typ or "", typ or "")

    def start_new_chat(self):
        """Initialize a new chat session."""
        # Clear conversation view
        for widget in self.conv_frame.winfo_children():
            widget.destroy()
        # Create a new chat ID and name
        chat_id = uuid.uuid4().hex
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_name = f"chat_{timestamp_str}"
        self.current_chat_id = chat_id
        # Append new chat to history and update listbox
        self.chats[chat_id] = chat_name
        self._refresh_history_list()
        # Prepare initial state for agent interaction
        self.current_state = {
            "user_input": None,
            "schema": None,
            "merged_input": None,
            "clarification_required": None,
            "questions": None,
            "sql_query": None,
            "query_result": None,
            "analytics": None,
            "viz_code": None,
            "sql_error": None,
            "sql_fix_attempts": 0,
            "chat_id": chat_id,
            "chat_name": chat_name,
            "step_num": 0,
        }
        # Inform the user
        self.append_message("Новый чат начат. Введите запрос.")
        # Enable send button
        self.send_btn.config(state=tk.NORMAL)

    def send_message(self):
        """Handle the Send button for a new chat session."""
        # If no chat session active, ignore
        if self.current_state is None or self.current_chat_id is None:
            return
        user_text = self.input_var.get().strip()
        if not user_text:
            return
        self.input_var.set("")
        # Display user's message in the conversation view
        self.append_message(f"Вы: {user_text}")
        # Offload agent processing to a thread to avoid blocking UI
        threading.Thread(target=self._process_user_input, args=(user_text,), daemon=True).start()

    def _process_user_input(self, user_text: str):
        """Run the agent logic in a background thread for responsiveness."""
        if not AGENT_AVAILABLE:
            # If agent functions are not available, simulate a response
            self.append_message("[Симуляция] Агент не доступен в этой среде.")
            return
        try:
            # Initial user request
            if self.current_state["user_input"] is None:
                # Save the user input
                self.current_state["user_input"] = user_text
                # log initial prompt and update state
                test.log_to_minio(self.current_state, "initial_prompt", user_text)
                # run schema introspection
                result = test.schema_introspection_node(self.current_state)
                self.current_state.update(result)
                # run intent detection
                result = test.intent_node(self.current_state)
                self.current_state.update(result)
                if self.current_state.get("clarification_required"):
                    qs = self.current_state.get("questions", [])
                    for q in qs:
                        self.append_message(f"Уточнение: {q}")
                    # prompt user to enter clarification; UI will handle next send
                else:
                    # proceed to SQL and viz
                    self._execute_sql_and_viz()
            else:
                # This is a clarification response
                # set merged input and log response
                base = self.current_state.get("merged_input") or self.current_state["user_input"]
                merged = f"{base}\n\nУТОЧНЕНИЕ:\n{user_text}"
                test.log_to_minio(self.current_state, "clarification_response", user_text)
                self.current_state["merged_input"] = merged
                # run intent again after clarification
                result = test.intent_node(self.current_state)
                self.current_state.update(result)
                if self.current_state.get("clarification_required"):
                    qs = self.current_state.get("questions", [])
                    for q in qs:
                        self.append_message(f"Уточнение: {q}")
                    # wait for further clarification
                else:
                    # proceed to SQL and viz
                    self._execute_sql_and_viz()
        except Exception as e:
            self.append_message(f"Ошибка агента: {e}")

    def _execute_sql_and_viz(self):
        """Execute SQL generation, run query, plan visualization and display results."""
        try:
            # Step 1: SQL generation based on current user intent
            result = test.sql_generation_node(self.current_state)
            self.current_state.update(result)

            # Step 2: SQL execution with automatic error fixing loop
            # Execute query and, if needed, attempt to fix errors up to 3 times
            while True:
                exec_result = test.sql_exec_node(self.current_state)
                self.current_state.update(exec_result)
                # If no error, break to next stage
                if self.current_state.get("sql_error") is None:
                    break
                # If error exists and no more attempts left, inform user and stop
                attempts = self.current_state.get("sql_fix_attempts", 0)
                if attempts >= 3:
                    self.append_message(f"SQL ошибка (не удалось исправить): {self.current_state['sql_error']}")
                    return
                # Otherwise, attempt to fix SQL
                fix_result = test.sql_error_fix_node(self.current_state)
                self.current_state.update(fix_result)
                # Loop back to re-execute

            # Step 3: Visualization planning
            viz_plan = test.viz_planning_node(self.current_state)
            self.current_state.update(viz_plan)
            # Display analytics immediately
            analytics = viz_plan.get("analytics")
            if analytics:
                self.append_message(f"Аналитика:\n{analytics}")
            # Step 4: Visualization execution
            viz_exec = test.viz_exec_node(self.current_state)
            self.current_state.update(viz_exec)
            # After viz_exec_node, an image link is logged in MinIO. Fetch and display the image.
            step_num = self.current_state.get("step_num")
            chat_id = self.current_state.get("chat_id")
            if step_num and chat_id and PIL_AVAILABLE and self.minio_client is not None:
                image_obj = f"{chat_id}/{step_num}_viz.png"
                try:
                    resp = self.minio_client.get_object(self.bucket_name, image_obj)
                    image_bytes = resp.read()
                    image_data = Image.open(io.BytesIO(image_bytes))
                    self.append_message("Визуализация:", image=image_data)
                except Exception:
                    pass
        except Exception as e:
            self.append_message(f"Ошибка обработки: {e}")
        finally:
            # After successful completion, disable send button; user may start a new chat
            self.send_btn.config(state=tk.DISABLED)

    def append_message(self, text: str, image: Optional['Image.Image'] = None):
        """Append a message or image to the conversation frame."""
        def _append():
            if text:
                lbl = tk.Label(self.conv_frame, text=text, bg="white", justify=tk.LEFT, anchor="w", wraplength=600)
                lbl.pack(fill=tk.X, anchor="w", padx=10, pady=2)
            if image is not None and PIL_AVAILABLE:
                # Resize image to fit, using a resampling method compatible
                # with Pillow versions >=10 where ANTIALIAS has been removed.
                max_width = 600
                w, h = image.size
                scale = min(1.0, max_width / w)
                new_size = (int(w * scale), int(h * scale))
                from PIL import Image as _PILImage  # Import inside to avoid issues when PIL is unavailable
                # Determine the appropriate resampling filter
                if hasattr(_PILImage, "Resampling"):
                    resample = _PILImage.Resampling.LANCZOS
                else:
                    # Fallbacks for older Pillow versions
                    resample = getattr(_PILImage, "LANCZOS", getattr(_PILImage, "BICUBIC", 1))
                try:
                    img_resized = image.resize(new_size, resample)
                except Exception:
                    # If resizing fails, use original image
                    img_resized = image
                photo = ImageTk.PhotoImage(img_resized)
                img_label = tk.Label(self.conv_frame, image=photo, bg="white")
                img_label.image = photo  # keep a reference to prevent garbage collection
                img_label.pack(anchor="w", padx=10, pady=5)
            # Auto-scroll to bottom
            self.canvas.update_idletasks()
            self.canvas.yview_moveto(1.0)
        self.after(0, _append)


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()