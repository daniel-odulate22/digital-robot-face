#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elzi Face — full renderer with mic-driven mouth, VAD, and TCP control
- SPACE: toggle 'talk' (synthetic)
- M: toggle microphone mode on/off
- B: blink
- 1..7: expressions (neutral, happy, sad, angry, surprised, sassy, sleepy)
- H/S/N: quick emotions (happy/sad/neutral)
- Arrows: gaze nudge; R: center gaze; [ ]: iris size; +/-: scale
- Q / ESC: quit

TCP control (port 8765):
  {"cmd":"talk","value":true|false}
  {"cmd":"blink"}
  {"cmd":"set_expression","value":"happy|sad|angry|surprised|sassy|sleepy|neutral"}
  {"cmd":"emotion","value":"happy|sad|neutral"}
  {"cmd":"set_gaze","x":-1..1,"y":-1..1}
  {"cmd":"mic","value":true|false}

Dependencies:
  pip install pygame sounddevice numpy
  # If sounddevice fails on your OS, the app still runs (mic disabled)
"""
import json
import math
import random
import socketserver
import threading
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import pygame

# ---- Optional mic stack (fails gracefully) ----
MIC_AVAILABLE = True
try:
    import sounddevice as sd
    import numpy as np
except Exception as e:
    MIC_AVAILABLE = False
    sd = None
    np = None
    print("[Elzi] Microphone stack unavailable:", e)

Vec2 = Tuple[float, float]
Color = Tuple[int, int, int]

# ---------- Utilities ----------
def clamp(v, lo, hi): return max(lo, min(hi, v))
def lerp(a, b, t): return a + (b - a) * t
def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def ease_toward(current: float, target: float, dt: float, tau: float) -> float:
    """Critically-damped-ish exponential approach using time constant 'tau' seconds."""
    if tau <= 0:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * alpha

# ---------- Parameters & State ----------
@dataclass
class FaceParams:
    # Colors
    bg: Color = (10, 12, 22)
    face: Color = (16, 18, 30)
    outline: Color = (40, 45, 70)
    eye_white: Color = (235, 240, 255)
    iris_col: Color = (75, 140, 255)
    pupil_col: Color = (10, 15, 24)
    mouth_col: Color = (240, 80, 110)
    accent: Color = (120, 180, 255)

    # Layout
    width: int = 1280
    height: int = 720
    scale: float = 1.0

    # Eyes
    eye_spacing: float = 260.0
    eye_radius: float = 120.0
    iris_radius: float = 46.0
    pupil_radius: float = 22.0

    # Lids
    eyelid_open: float = 1.0
    lowerlid_open: float = 0.05

    # Mouth
    mouth_open: float = 0.12        # visual state 0..1
    mouth_width: float = 220.0
    mouth_thickness: int = 8
    mouth_tau: float = 0.07          # response

    # Dynamics
    blink_interval_range: Tuple[float, float] = (2.5, 6.0)
    blink_duration: float = 0.12
    saccade_interval_range: Tuple[float, float] = (0.7, 2.5)
    gaze_tau: float = 0.08
    eyelid_tau: float = 0.06

@dataclass
class FaceState:
    time: float = 0.0
    expression: str = "neutral"
    emotion: str = "neutral"
    talk: bool = False

    # eyes / gaze
    mouse_follow: bool = True
    gaze: Vec2 = (0.0, 0.0)
    gaze_target: Vec2 = (0.0, 0.0)
    next_saccade_in: float = 0.0

    # blink
    blink_t: float = -1.0            # -1 means idle; otherwise 0..1
    next_blink_in: float = 0.0

    # mic
    mic_enabled: bool = False
    speaking_gate: bool = False      # VAD state (attack/release)
    vad_hold: float = 0.0            # release hold timer

# ---------- Microphone helper ----------
class MicInput:
    """
    Lightweight mic reader using sounddevice. Pull-based.
    Each call to .read_rms(duration) records a short chunk and returns RMS in 0..1.
    """
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        if not MIC_AVAILABLE:
            raise RuntimeError("Mic stack not available")
        self.samplerate = samplerate
        self.channels = channels

    def read_rms(self, duration: float = 0.08) -> float:
        # record small chunk; convert to float32; compute RMS; normalize and clamp
        frames = int(duration * self.samplerate)
        try:
            audio = sd.rec(frames, samplerate=self.samplerate, channels=self.channels, dtype="float32", blocking=True)
            # audio shape: (frames, 1)
            rms = float(np.sqrt(np.mean(np.square(audio))))
            # Scale factor: tweak to taste to reach ~0..1 for normal speech
            scaled = rms * 8.0
            return clamp(scaled, 0.0, 1.0)
        except Exception as e:
            # On error, act like silence so UI still runs
            # print("Mic read error:", e)
            return 0.0

# ---------- Main Face ----------
class ElziFace:
    def __init__(self, params: FaceParams):
        self.p = params
        self.s = FaceState()
        self._rng = random.Random()
        self._schedule_blink()
        self._schedule_saccade()
        self._set_expression_defaults("neutral")

        # mic
        self.mic: Optional[MicInput] = None
        if MIC_AVAILABLE:
            try:
                self.mic = MicInput()
            except Exception as e:
                print("[Elzi] Mic init failed, continuing without mic:", e)
                self.mic = None

        # smoothing buffers
        from collections import deque
        self._vol_hist = deque(maxlen=6)   # rolling average for stability

    # ---- Expressions ----
    def _expressions(self):
        return {
            "neutral":   dict(upper=1.0, lower=0.05, mouth=0.12, iris=self.p.iris_radius),
            "happy":     dict(upper=1.0, lower=0.20, mouth=0.35),
            "sad":       dict(upper=0.7, lower=0.10, mouth=0.08),
            "angry":     dict(upper=0.55, lower=0.10, mouth=0.18),
            "surprised": dict(upper=1.0, lower=0.00, mouth=0.85, iris=self.p.iris_radius*0.92),
            "sassy":     dict(upper=0.9, lower=0.05, mouth=0.22),
            "sleepy":    dict(upper=0.25, lower=0.00, mouth=0.06),
        }

    def set_expression(self, name: str):
        name = (name or "").lower()
        if name in self._expressions():
            self.s.expression = name
            self._set_expression_defaults(name)

    def _set_expression_defaults(self, name: str):
        e = self._expressions()[name]
        self.p.eyelid_open = e.get("upper", self.p.eyelid_open)
        self.p.lowerlid_open = e.get("lower", self.p.lowerlid_open)
        self.p.mouth_open = e.get("mouth", self.p.mouth_open)
        if "iris" in e:
            self.p.iris_radius = e["iris"]

    # ---- Schedules ----
    def _schedule_blink(self):
        self.s.next_blink_in = self._rng.uniform(*self.p.blink_interval_range)

    def _schedule_saccade(self):
        self.s.next_saccade_in = self._rng.uniform(*self.p.saccade_interval_range)

    # ---- Public actions ----
    def talk(self, on: bool):
        self.s.talk = bool(on)

    def blink(self):
        if self.s.blink_t < 0:
            self.s.blink_t = 0.0

    # ---- Update / Logic ----
    def update(self, dt: float, mouse_pos: Vec2, center: Vec2):
        cx, cy = center
        self.s.time += dt

        # Gaze target (mouse follow + micro-saccades)
        if self.s.mouse_follow:
            mx, my = mouse_pos
            self.s.gaze_target = (
                clamp((mx - cx) / (self.p.width * 0.30), -1, 1),
                clamp((cy - my) / (self.p.height * 0.30), -1, 1)
            )

        self.s.next_saccade_in -= dt
        if self.s.next_saccade_in <= 0:
            gx, gy = self.s.gaze_target
            self.s.gaze_target = (
                clamp(gx + self._rng.uniform(-0.15, 0.15), -1, 1),
                clamp(gy + self._rng.uniform(-0.10, 0.10), -1, 1)
            )
            self._schedule_saccade()

        self.s.gaze = (
            ease_toward(self.s.gaze[0], self.s.gaze_target[0], dt, self.p.gaze_tau),
            ease_toward(self.s.gaze[1], self.s.gaze_target[1], dt, self.p.gaze_tau),
        )

        # Blink logic
        if self.s.blink_t >= 0:
            self.s.blink_t += dt / self.p.blink_duration
            if self.s.blink_t >= 1:
                self.s.blink_t = -1
                self._schedule_blink()
        else:
            self.s.next_blink_in -= dt
            if self.s.next_blink_in <= 0:
                self.s.blink_t = 0.0

        # --- Mouth target from mic OR synthetic talk ---
        mouth_target = self._mouth_target_from_mic(dt)
        if mouth_target is None:
            # fallback to synthetic talk oscillation if talk==True
            mouth_target = self._mouth_target_from_talk(dt)

        # Emotion influence (subtle tilt up/down via open amount bias)
        emotion_offset = {
            "happy": 0.15,
            "sad":  -0.10,
        }.get(self.s.emotion, 0.0)

        # final open target
        mouth_target = clamp((mouth_target or 0.0) + emotion_offset, 0.0, 1.0)
        self.p.mouth_open = ease_toward(self.p.mouth_open, mouth_target, dt, self.p.mouth_tau * 0.8)

    def _mouth_target_from_talk(self, dt: float) -> float:
        if not self.s.talk:
            # gently close when not talking
            return clamp(self.p.mouth_open * 0.7, 0.02, 1.0)
        # Smooth semi-circle oscillation while talking (pleasant motion curve)
        phase = (math.sin(self.s.time * 7.5) * 0.5 + 0.5)  # 0..1
        semi = math.sqrt(max(0.0, 1.0 - (phase - 1.0) ** 2))  # semicircle arc
        jitter = self._rng.random() * 0.08
        return clamp(self.p.mouth_open * 0.4 + semi * 0.6 + jitter * 0.15, 0.04, 1.0)

    def _mouth_target_from_mic(self, dt: float) -> Optional[float]:
        """Return mouth target 0..1 from mic+VAD if mic mode is ON, else None."""
        if not self.s.mic_enabled or self.mic is None:
            return None

        vol = self.mic.read_rms(0.06)  # 60ms chunk
        self._vol_hist.append(vol)
        smooth_vol = sum(self._vol_hist) / max(1, len(self._vol_hist))

        # Simple VAD with threshold + attack/release
        THRESH = 0.035     # energy threshold; tweak per environment
        ATTACK = 0.02      # seconds to latch into speaking
        RELEASE = 0.15     # seconds to hold after falling below threshold

        if smooth_vol > THRESH:
            self.s.speaking_gate = True
            self.s.vad_hold = RELEASE
        else:
            # release with hold timeout
            self.s.vad_hold -= dt
            if self.s.vad_hold <= 0:
                self.s.speaking_gate = False

        if self.s.speaking_gate:
            # map volume to mouth open; add light compression to keep natural
            target = clamp( (smooth_vol - THRESH) / (1.0 - THRESH), 0.0, 1.0 )
            # mild nonlinear boost for clarity
            target = clamp(math.pow(target, 0.7) * 1.1, 0.0, 1.0)
            # minimum aperture so it feels alive
            target = clamp(target * 0.95 + 0.05, 0.0, 1.0)
            return target
        else:
            # idle closed
            return 0.03

    # ---- Draw ----
    def draw(self, surf: pygame.Surface):
        surf.fill(self.p.bg)
        cx, cy = self.p.width // 2, self.p.height // 2

        # Face panel
        plate = pygame.Rect(0, 0, int(self.p.width * 0.72 * self.p.scale), int(self.p.height * 0.62 * self.p.scale))
        plate.center = (cx, cy + 8)
        pygame.draw.rect(surf, self.p.face, plate, border_radius=40)
        pygame.draw.rect(surf, self.p.outline, plate, width=2, border_radius=40)

        # Eyes
        eye_o = self.p.eye_spacing * self.p.scale * 0.5
        positions = [(cx - eye_o, cy - 40), (cx + eye_o, cy - 40)]
        for (ex, ey) in positions:
            eye_rect = pygame.Rect(0, 0, int(self.p.eye_radius * 2.4 * self.p.scale), int(self.p.eye_radius * 1.8 * self.p.scale))
            eye_rect.center = (ex, ey)
            pygame.draw.ellipse(surf, self.p.eye_white, eye_rect)
            pygame.draw.ellipse(surf, self.p.outline, eye_rect, width=2)

            # Blink mask amounts
            open_u = self.p.eyelid_open
            open_l = self.p.lowerlid_open
            if self.s.blink_t >= 0:
                t = self.s.blink_t
                # closing then opening curve with smoothstep
                k = (1 - smoothstep(t * 2)) if t <= 0.5 else smoothstep((t - 0.5) * 2)
                open_u = min(open_u, k)
                open_l = min(open_l + (1 - k) * 0.4, 1.0)

            lid_th = int(eye_rect.height * (1 - open_u) * 0.85)
            if lid_th > 0:
                pygame.draw.rect(surf, self.p.face, (eye_rect.left, eye_rect.top, eye_rect.width, lid_th))
            low_th = int(eye_rect.height * open_l * 0.55)
            if low_th > 0:
                pygame.draw.rect(surf, self.p.face, (eye_rect.left, eye_rect.bottom - low_th, eye_rect.width, low_th))

            # Iris + pupil with gaze offset
            gpx = self.s.gaze[0] * self.p.eye_radius * 0.35 * self.p.scale
            gpy = -self.s.gaze[1] * self.p.eye_radius * 0.35 * self.p.scale
            ix, iy = int(ex + gpx), int(ey + gpy)
            pygame.draw.circle(surf, self.p.iris_col, (ix, iy), int(self.p.iris_radius * self.p.scale))
            pygame.draw.circle(surf, self.p.pupil_col, (ix, iy), int(self.p.pupil_radius * self.p.scale))
            # highlight
            pygame.draw.circle(
                surf, (255, 255, 255),
                (int(ix - self.p.iris_radius * self.p.scale * 0.35), int(iy - self.p.iris_radius * self.p.scale * 0.35)),
                max(2, int(self.p.iris_radius * self.p.scale * 0.22))
            )

        # --- Mouth: Semi-circle arc, width responds to mouth_open ---
        mx, my = cx, cy + int(plate.height * 0.18)
        mw = int(self.p.mouth_width * self.p.scale)
        # height = visual open; arc rect height scales with mouth_open
        mh = int(lerp(8, 110 * self.p.scale, clamp(self.p.mouth_open, 0.0, 1.0)))
        mouth_rect = pygame.Rect(0, 0, mw, max(8, mh))
        mouth_rect.center = (mx, my)

        # emotion tilt subtly narrows/widens the arc angles
        emo_tilt = (0.15 if self.s.emotion == "happy" else -0.10 if self.s.emotion == "sad" else 0.0)
        start = math.pi + emo_tilt
        end = 2 * math.pi - emo_tilt
        pygame.draw.arc(surf, self.p.mouth_col, mouth_rect, start, end, self.p.mouth_thickness)

        # Accent glow bar
        glow = pygame.Rect(0, 0, plate.width, 10)
        glow.midtop = (plate.centerx, plate.bottom - 18)
        pygame.draw.rect(surf, self.p.accent, glow, border_radius=10)

        # HUD (mic state)
        font = pygame.font.SysFont(None, 22)
        mic_txt = f"MIC [{'ON' if (self.s.mic_enabled and self.mic is not None) else 'OFF'}]"
        vad_txt = f"VAD [{'SPEAK' if self.s.speaking_gate else '----'}]"
        hud1 = font.render(mic_txt, True, (200, 220, 255))
        hud2 = font.render(vad_txt, True, (200, 220, 255))
        surf.blit(hud1, (12, 10))
        surf.blit(hud2, (12, 32))

    # ---- Commands from TCP ----
    def apply_command(self, cmd: dict):
        c = (cmd.get("cmd") or "").lower()
        if c == "set_expression":
            self.set_expression(cmd.get("value", "neutral"))
        elif c == "talk":
            self.talk(bool(cmd.get("value", True)))
        elif c == "blink":
            self.blink()
        elif c == "emotion":
            self.s.emotion = (cmd.get("value") or "neutral").lower()
        elif c == "set_gaze":
            x = float(cmd.get("x", 0.0)); y = float(cmd.get("y", 0.0))
            self.s.gaze_target = (clamp(x, -1.0, 1.0), clamp(y, -1.0, 1.0))
        elif c == "mic":
            want = bool(cmd.get("value", True))
            if self.mic is None and want:
                print("[Elzi] Mic backend not available on this system.")
            else:
                self.s.mic_enabled = want
                print(f"[Elzi] Microphone {'enabled' if want else 'disabled'}")

# ---------- TCP control server ----------
class _TCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline().decode().strip()
            if not line:
                self.wfile.write(b"error: empty\n")
                return
            data = json.loads(line)
            self.server.queue.append(data)
            self.wfile.write(b"ok\n")
        except Exception as e:
            self.wfile.write(f"error: {e}\n".encode())

class FaceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    def __init__(self, addr, Handler):
        super().__init__(addr, Handler)
        self.queue = []

# ---------- Main loop ----------
def main():
    pygame.init()
    pygame.display.set_caption("Elzi Face")
    p = FaceParams()
    screen = pygame.display.set_mode((p.width, p.height))
    clock = pygame.time.Clock()

    face = ElziFace(p)

    # TCP server
    server = FaceServer(("127.0.0.1", 8765), _TCPHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("[Elzi] TCP control on 127.0.0.1:8765")

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    face.talk(not face.s.talk)
                elif k == pygame.K_f:
                    face.s.mouse_follow = not face.s.mouse_follow
                    print(f"Mouse follow {'enabled' if face.s.mouse_follow else 'disabled'}")

                elif k == pygame.K_b:
                    face.blink()
                elif k == pygame.K_m:
                    if face.mic is None:
                        print("[Elzi] Mic backend not available on this system.")
                    else:
                        face.s.mic_enabled = not face.s.mic_enabled
                        print(f"[Elzi] Microphone {'enabled' if face.s.mic_enabled else 'disabled'}")
                        # When mic turns on, clear history for clean ramp
                        face._vol_hist.clear()
                elif k == pygame.K_r:
                    face.s.gaze_target = (0.0, 0.0)
                elif k == pygame.K_LEFT:
                    gx, gy = face.s.gaze_target; face.s.gaze_target = (clamp(gx - 0.05, -1, 1), gy)
                elif k == pygame.K_RIGHT:
                    gx, gy = face.s.gaze_target; face.s.gaze_target = (clamp(gx + 0.05, -1, 1), gy)
                elif k == pygame.K_UP:
                    gx, gy = face.s.gaze_target; face.s.gaze_target = (gx, clamp(gy + 0.05, -1, 1))
                elif k == pygame.K_DOWN:
                    gx, gy = face.s.gaze_target; face.s.gaze_target = (gx, clamp(gy - 0.05, -1, 1))
                elif k == pygame.K_LEFTBRACKET:
                    p.iris_radius = clamp(p.iris_radius - 2, 22, 80)
                elif k == pygame.K_RIGHTBRACKET:
                    p.iris_radius = clamp(p.iris_radius + 2, 22, 80)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS):
                    p.scale = clamp(p.scale + 0.05, 0.6, 1.6)
                elif k == pygame.K_MINUS:
                    p.scale = clamp(p.scale - 0.05, 0.6, 1.6)

                # Quick emotions
                elif k == pygame.K_h:
                    face.s.emotion = "happy"
                elif k == pygame.K_s:
                    face.s.emotion = "sad"
                elif k == pygame.K_n:
                    face.s.emotion = "neutral"

                # Expressions 1..7
                elif k == pygame.K_1:
                    face.set_expression("neutral")
                elif k == pygame.K_2:
                    face.set_expression("happy")
                elif k == pygame.K_3:
                    face.set_expression("sad")
                elif k == pygame.K_4:
                    face.set_expression("angry")
                elif k == pygame.K_5:
                    face.set_expression("surprised")
                elif k == pygame.K_6:
                    face.set_expression("sassy")
                elif k == pygame.K_7:
                    face.set_expression("sleepy")

        # Pump queued TCP commands
        while server.queue:
            face.apply_command(server.queue.pop(0))

        face.update(dt, pygame.mouse.get_pos(), (p.width // 2, p.height // 2))
        face.draw(screen)
        pygame.display.flip()

    server.shutdown()
    pygame.quit()

if __name__ == "__main__":
    main()
