import gradio as gr
import os
import json
import time
import subprocess
from glob import glob

# --- PATH SETUP ---
HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "voicechat2", "data", "characters")
CONFIG_PATH = os.path.join(HOME, "voicechat2", "active_config.json")
os.makedirs(BASE_DIR, exist_ok=True)

# --- LOGIC FUNCTIONS ---

def list_characters():
    try:
        return sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    except: return []

def list_logs(char):
    if not char: return []
    try:
        log_pattern = os.path.join(BASE_DIR, char, "logs", "*.json")
        files = glob(log_pattern)
        files.sort(key=os.path.getmtime, reverse=True)
        return [os.path.basename(f) for f in files]
    except: return []

def load_log_content(char, log_file):
    if not char or not log_file: return []
    path = os.path.join(BASE_DIR, char, "logs", log_file)
    rows = []
    try:
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    user_val = item.get('instruction') or item.get('user') or ""
                    bot_val = item.get('output') or item.get('assistant') or ""
                    rows.append([idx, user_text, bot_val])
        return rows
    except: return []

def update_kb_labels(dtype):
    if dtype == "Lore":
        return (gr.update(label="Fact Topic / Subject", placeholder="e.g. Origin"),
                gr.update(label="Factual Information", placeholder="e.g. Built in Fort Worth..."))
    elif dtype == "Prose":
        return (gr.update(label="Narrative Prompt / Mood", placeholder="e.g. Existential thought"),
                gr.update(label="Character's Internal Monologue", placeholder="e.g. The sky is blue..."))
    elif dtype == "Social":
        return (gr.update(label="Other Character's Line", placeholder="e.g. Unit 02: Morning."),
                gr.update(label="Character's Reaction", placeholder="e.g. Bones: Is it?"))
    return gr.update(label="Instruction"), gr.update(label="Response")

def commit_log_to_training(char, df_data):
    if not char or df_data is None: return "❌ Select character", df_data
    try:
        if hasattr(df_data, "values"): data_list = df_data.values.tolist()
        elif isinstance(df_data, dict) and "data" in df_data: data_list = df_data["data"]
        else: data_list = df_data
        train_path = os.path.join(BASE_DIR, char, "conversational.json")
        master_data = json.load(open(train_path, "r")) if os.path.exists(train_path) else []
        added = 0
        for row in data_list:
            if len(row) >= 3 and str(row[1]).strip() and str(row[2]).strip():
                master_data.append({"instruction": str(row[1]), "output": str(row[2]), "type": "conversational_log", "ts": time.time()})
                added += 1
        with open(train_path, "w") as f: json.dump(master_data, f, indent=2)
        return f"✅ Saved {added} entries to {char}.", []
    except Exception as e: return f"❌ Error: {str(e)}", df_data

def save_manual_data(char, dtype, instruction, output):
    if not char: return "❌ Select a character first.", instruction, output
    file_path = os.path.join(BASE_DIR, char, f"{dtype.lower()}.json")
    try:
        data = json.load(open(file_path, "r")) if os.path.exists(file_path) else []
        data.append({"instruction": instruction, "output": output, "type": dtype.lower(), "ts": time.time()})
        with open(file_path, "w") as f: json.dump(data, f, indent=2)
        return f"✅ {dtype} Recorded!", "", ""
    except Exception as e: return f"❌ Error: {str(e)}", instruction, output

def create_new_character(name):
    if not name: return "Enter a name."
    clean_name = name.lower().strip().replace(" ", "_")
    path = os.path.join(BASE_DIR, clean_name)
    if os.path.exists(path): return "Character exists."
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)
    for dtype in ["lore.json", "conversational.json", "prose.json", "social.json"]:
        with open(os.path.join(path, dtype), "w") as f: json.dump([], f)
    return f"✅ Created {clean_name}."

def hot_swap_character(name):
    if not name: return "Select a character."
    with open(CONFIG_PATH, "w") as f:
        json.dump({"active_character": name, "ts": time.time()}, f)
    return f"🚀 Swapped Live Playback to: {name}"

def run_universal_training(char):
    if not char: 
        yield "❌ Error: Select a character first!"
        return
    
    yield f"🏗️ Starting Forge for {char}... Stopping VRAM processes...\n"
    
    # Run the universal script as a subprocess
    process = subprocess.Popen(['python', 'lychee_forge.py', char], 
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output = ""
    for line in process.stdout:
        output += line
        yield output

# --- UI DEFINITION ---

with gr.Blocks(title="Lychee Factory") as demo:
    gr.Markdown("# 🦿 Lychee Character Factory")
    
    with gr.Tab("1. Log Review"):
        with gr.Row():
            char_drop = gr.Dropdown(choices=list_characters(), label="Character")
            log_drop = gr.Dropdown(choices=[], label="Session Log")
            refresh_btn = gr.Button("🔄 Sync Disk", variant="secondary", scale=0)
        chat_editor = gr.Dataframe(headers=["ID", "User", "Assistant"], datatype=["number", "str", "str"], interactive=True)
        commit_btn = gr.Button("📥 Commit Edits", variant="primary")
        status_logs = gr.Textbox(label="Status")
        
        refresh_btn.click(lambda: (gr.Dropdown(choices=list_characters(), value=None), gr.Dropdown(choices=[], value=None), []), outputs=[char_drop, log_drop, chat_editor])
        char_drop.change(lambda c: gr.Dropdown(choices=list_logs(c), value=None), inputs=char_drop, outputs=log_drop)
        log_drop.change(load_log_content, inputs=[char_drop, log_drop], outputs=chat_editor)
        commit_btn.click(commit_log_to_training, inputs=[char_drop, chat_editor], outputs=[status_logs, chat_editor])

    with gr.Tab("2. Knowledge Base"):
        with gr.Row():
            kb_char = gr.Dropdown(choices=list_characters(), label="Character")
            kb_type = gr.Radio(["Lore", "Prose", "Social"], label="Data Type", value="Lore")
        kb_inst = gr.Textbox(label="Fact Topic / Subject")
        kb_out = gr.Textbox(label="Factual Information", lines=5)
        kb_btn = gr.Button("💾 Save Entry", variant="primary")
        kb_status = gr.Textbox(label="Status")
        kb_type.change(update_kb_labels, inputs=kb_type, outputs=[kb_inst, kb_out])
        kb_btn.click(save_manual_data, inputs=[kb_char, kb_type, kb_inst, kb_out], outputs=[kb_status, kb_inst, kb_out])

    with gr.Tab("3. Character Creator"):
        new_name = gr.Textbox(label="New Character ID")
        create_btn = gr.Button("Initialize Structure")
        c_out = gr.Textbox(label="Result")
        create_btn.click(create_new_character, inputs=new_name, outputs=c_out)

    with gr.Tab("4. The Lab"):
        gr.Markdown("### 🚀 Hot-Swap Model")
        swap_char = gr.Dropdown(choices=list_characters(), label="Target Character")
        swap_btn = gr.Button("Switch Live Model Now", variant="primary")
        swap_status = gr.Textbox(label="Swap Result")
        swap_btn.click(hot_swap_character, inputs=swap_char, outputs=swap_status)
        
        gr.Markdown("---")
        gr.Markdown("### 🔥 Universal Forge\n## ⚠️ VRAM WARNING: Stop all voice servers first!")
        train_char = gr.Dropdown(choices=list_characters(), label="Character to Train")
        train_btn = gr.Button("🚀 Start Universal Training Run", variant="stop")
        train_out = gr.Code(label="Forge Output")
        train_btn.click(run_universal_training, inputs=train_char, outputs=train_out)

if __name__ == "__main__":
    demo.launch(server_port=7860, theme=gr.themes.Soft())