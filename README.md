# Digital Robot Face — Animated AI Companion UI (Python + Pygame)

The robot face is the **visual personality layer** for Elzi — a Jarvis-inspired AI companion.  
It’s an animated face built in Python with `pygame` that reacts to mic input, keyboard toggles, and interactions in real-time.
<img width="988" height="499" alt="robot-face" src="https://github.com/user-attachments/assets/4918cf18-9f9a-4cb4-adab-68695058c16b" />

---

## ✨ Features
- **Talking Mouth** — Semi-circle mouth movement synced with microphone input.  
- **Live Mic Mode** — Toggle mic input on/off.  
- **Expressions** — Switch between emotions like happy, sad, surprised, neutral.  
- **Eye Tracking** — Eyes follow your mouse (toggleable).  
- **Blinking & Random Gaze** for realism.  
- **Keyboard Shortcuts** for easy control.  

---

## 🎹 Controls
| Key | Action |
|-----|---------|
| `SPACE` | Toggle talking mode |
| `M` | Toggle mic input on/off |
| `F` | Toggle mouse follow on/off |
| `B` | Blink |
| `H` | Set Happy expression |
| `S` | Set Sad expression |
| `N` | Set Neutral expression |
| `1..7` | Quick expressions (neutral, happy, sad, angry, surprised, sassy, sleepy) |
| `ESC` / `Q` | Quit program |

---

## 🛠️ Installation

1. **Clone this repo (or download the script):**
   ```bash
   git clone https://github.com/your-username/elzi-face.git
   cd elzi-face
2. **Create a virtual environment (optinal but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
3. **Install dependencies**
   ```bash
     pip install pygame sounddevice numpy
4. **RUN**
    ```bash
      python elzi_face.py
