from collections import Counter, deque
import random
import time

import cv2
import mediapipe as mp
import numpy as np

W, H = 960, 540
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------- 규칙 / 타이밍 설정 ----------
RPS_WEIGHTS = (1, 1, 1)
CHAM_WEIGHTS = (1, 1)
BEAT = 0.7
CAP_START = 0.25
CAP_LEN = 0.6
READY_HOLD = 0.5
RESULT_T = 1.8
MIN_SAMPLES = 4
DEAD = 40

CAP0 = 2 * BEAT + CAP_START
CAP1 = CAP0 + CAP_LEN

PLAYER_COL, CPU_COL = (80, 210, 80), (70, 70, 235)
GOLD, GRAY, WHITE = (60, 200, 255), (160, 160, 160), (255, 255, 255)
OWNER_COL = {"player": PLAYER_COL, "cpu": CPU_COL}
KIND_COL = {
    "clear": PLAYER_COL,
    "win": PLAYER_COL,
    "tie": GOLD,
    "continue": GOLD,
    "invalid": GOLD,
    "restart": CPU_COL,
    "lose": CPU_COL,
}


def S(ko, en):
  return en


TEXTS = []


def T(text, x, y, size=28, color=WHITE, anchor="la"):
  TEXTS.append((text, int(x), int(y), size, color, anchor))


def flush_texts(img):
  if not TEXTS:
    return img
  for text, x, y, size, col, anchor in TEXTS:
    scale = size / 32.0
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    ox = x - (tw // 2 if anchor[0] == "m" else tw if anchor[0] == "r" else 0)
    if anchor[1] == "m":
      oy = y + th // 2
    elif anchor[1] in "bsd":
      oy = y
    else:
      oy = y + th
    cv2.putText(
        img,
        text,
        (ox, oy),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (ox, oy),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        col,
        2,
        cv2.LINE_AA,
    )
  TEXTS.clear()
  return img


BEATS = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}
HAND_EN = {"rock": "ROCK", "paper": "PAPER", "scissors": "SCISSORS"}
LABELS = {
    0: (S("가위", "SCISSORS"), S("바위", "ROCK"), S("보", "PAPER")),
    1: (S("묵", "ROCK"), S("찌", "SCISSORS"), S("빠", "PAPER")),
    2: (S("참", "CHAM"), S("참", "CHAM"), S("참", "CHAM")),
}


def beats(a, b):
  return (a, b) in BEATS


def who(x):
  return S("플레이어", "YOU") if x == "player" else S("컴퓨터", "CPU")


def other(x):
  return "cpu" if x == "player" else "player"


def hname(stage, k):
  if k == "left":
    return S("왼쪽", "LEFT")
  if k == "right":
    return S("오른쪽", "RIGHT")
  return HAND_EN[k]


class Game:

  def __init__(self):
    self.reset()

  def reset(self):
    self.stage = 0  # 가위바위보: 0, 묵찌빠: 1, 참참참: 2
    self.owner = None  # 연승자
    self.attacker = None  # 묵찌빠 공격권
    self.over = None  # 최종 승리자

  # 연승이 끊겼을 때
  def _broken(self, stage_name):
    self.stage, self.owner, self.attacker = 0, None, None
    return "restart", S(
        f"{stage_name}에서 연승 끊김 — 처음부터!", "Streak broken - restart!"
    )

  # 연승 중
  def resolve(self, person, cpu):
    s = self.stage
    if s == 0:
      if person == cpu:
        return "tie", S("비겼습니다 — 다시!", "Tie - again!")
      win = "player" if beats(person, cpu) else "cpu"
      self.owner = self.attacker = win
      self.stage = 1  # 이기면 다음단계
      return "clear", S(
          f"{who(win)} 승! 다음: 묵찌빠", f"{who(win)} wins! Next: Mukjjippa"
      )

    if s == 1:
      if person != cpu:
        new = "player" if beats(person, cpu) else "cpu"
        msg = (
            S(f"공격 유지 — {who(new)}", f"Attack stays: {who(new)}")
            if new == self.attacker
            else S(f"공격권 → {who(new)}", f"Attack -> {who(new)}")
        )
        self.attacker = new
        return "continue", msg
      win = self.attacker
      if win == self.owner:
        self.stage = 2
        return "clear", S(
            f"묵찌빠 {who(win)} 승! 다음: 참참참",
            f"{who(win)} wins! Next: Chamchamcham",
        )
      return self._broken(S("묵찌빠", "Mukjjippa"))

    atk = self.owner
    win = atk if person == cpu else other(atk)
    if win == self.owner:
        self.over = win
        if win == "player":
          return "win", S("3연승! 플레이어 승리!", "3 in a row! YOU WIN!")
        return "lose", S("컴퓨터 3연승… 패배", "CPU wins 3 in a row")
    return self._broken(S("참참참", "Chamchamcham"))


# 손 인식
def classify_hand(pts):
  if pts is None or len(pts) < 21:
    return None
  d = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))
  try:
    ext = [
        d(0, t) > 1.1 * d(0, p)
        for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))
    ]
  except Exception:
    return None
  n = sum(ext)
  if n == 0:
    return "rock"
  if n == 4:
    return "paper"
  if ext[0] and ext[1] and not ext[2] and not ext[3]:
    return "scissors"
  return None


# 참참참 손바닥 위치 판단
def hand_side(pts):
  if pts is None or len(pts) < 21:
    return None
  try:
    cx = float(pts[[0, 5, 9, 13, 17]].mean(axis=0)[0])
  except Exception:
    return None
  if abs(cx - W / 2) < DEAD:
    return None
  return "left" if cx < W / 2 else "right"


# 플레이어 패 가져오기
def sample_of(stage, pts):
  if pts is None:
    return None
  return hand_side(pts) if stage == 2 else classify_hand(pts)


# CPU가 낼 값 가져오기
def pick(stage):
  if stage == 2:
    return random.choices(("left", "right"), weights=CHAM_WEIGHTS)[0]
  return random.choices(("scissors", "rock", "paper"), weights=RPS_WEIGHTS)[0]


def draw_ui(frame, g, phase, t, result, live_pose, pts):
  st = g.stage
  side = hand_side(pts)
  cx, cy = W // 2, H // 2

  if st == 2 and phase != "over":
    ov = frame.copy()
    if side == "left":
      cv2.rectangle(ov, (0, 0), (W // 2, H), (255, 170, 60), -1)
    elif side == "right":
      cv2.rectangle(ov, (W // 2, 0), (W, H), (60, 130, 255), -1)
    cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
    cv2.line(frame, (W // 2, 0), (W // 2, H), WHITE, 2, cv2.LINE_AA)
    for dx in (-DEAD, DEAD):
      cv2.line(
          frame,
          (W // 2 + dx, 0),
          (W // 2 + dx, H),
          (200, 200, 200),
          1,
          cv2.LINE_AA,
      )
    T(S("왼쪽", "LEFT"), W // 4, H - 110, 44, (255, 200, 120), "mm")
    T(S("오른쪽", "RIGHT"), 3 * W // 4, H - 110, 44, (120, 160, 255), "mm")

  owner = g.over or g.owner
  done = 3 if g.over else st
  names = [
      S("가위바위보", "RPS"),
      S("묵찌빠", "MUKJJIPPA"),
      S("참참참", "CHAMCHAMCHAM"),
  ]
  for i, nm in enumerate(names):
    x0 = 14 + i * 210
    if i < done:
      border, tcol = OWNER_COL[owner], OWNER_COL[owner]
    elif i == st:
      border, tcol = GOLD, WHITE
    else:
      border, tcol = GRAY, GRAY
    cv2.rectangle(frame, (x0, 14), (x0 + 200, 60), (35, 35, 35), -1)
    cv2.rectangle(frame, (x0, 14), (x0 + 200, 60), border, 3)
    T(nm, x0 + 100, 37, 24, tcol, "mm")

  if owner:
    T(
        S(f"연승: {who(owner)} {done}/3", f"Streak: {who(owner)} {done}/3"),
        14,
        68,
        26,
        OWNER_COL[owner],
    )
  else:
    T(S("연승 없음", "No streak"), 14, 68, 26, GRAY)
  if st == 1 and g.attacker and not g.over:
    T(
        S(f"공격권: {who(g.attacker)}", f"Attack: {who(g.attacker)}"),
        300,
        68,
        26,
        OWNER_COL[g.attacker],
    )

  if st == 0:
    hint = S(
        "이긴 쪽이 연승 시작 · 비기면 다시", "Winner starts streak · tie = replay"
    )
  elif st == 1:
    hint = S(
        "같은 손이면 공격 중인 쪽 승 · 다르면 공격권 이동",
        "Same hand: attacker wins",
    )
  else:
    hint = S(
        f"손 위치로 방향 선택 · 같으면 {who(owner)} 승",
        f"Hand side = direction · same: {who(owner)} wins",
    )
  T(hint, 14, 104, 20, GRAY)

  px0, py0, px1, py1 = W - 230, 14, W - 14, 170
  cv2.rectangle(frame, (px0, py0), (px1, py1), (35, 35, 35), -1)
  cv2.rectangle(frame, (px0, py0), (px1, py1), CPU_COL, 2)
  T(S("컴퓨터", "COMPUTER"), px0 + 14, py0 + 8, 24, CPU_COL)
  if phase in ("result", "over") and result and result[3]:
    c, rst = result[3], result[4]
    T(hname(rst, c), px0 + 118, py0 + 98, 34, WHITE, "mm")
  else:
    T("?", (px0 + px1) // 2, py0 + 98, 70, GRAY, "mm")

  if phase == "idle":
    T(S("손을 카메라에 보여주세요", "Show your hand to the camera"), cx, cy, 36, WHITE, "mm")
  elif phase == "count":
    idx = min(2, int(t / BEAT))
    T(LABELS[st][idx] + ("!" if idx == 2 else ""), cx, cy, 150, GOLD, "mm")
    if t >= CAP0:
      T(S("인식 중...", "reading..."), cx, cy + 110, 30, WHITE, "mm")
  elif result:
    kind, text, p, c, rst = result
    col = OWNER_COL[g.owner] if (kind == "clear" and g.owner) else KIND_COL[kind]
    T(text, cx, cy - 20, 38, col, "mm")
    if p:
      T(
          f"{S('나', 'YOU')}: {hname(rst, p)}   VS   {S('컴퓨터', 'CPU')}: {hname(rst, c)}",
          cx,
          cy + 45,
          30,
          WHITE,
          "mm",
      )

  if phase != "over":
    if pts is None:
      T(S("손이 보이지 않아요", "No hand"), 14, H - 14, 26, GRAY, "lb")
    elif st == 2:
      T(
          S("내 위치: ", "You: ")
          + (hname(2, side) if side else S("가운데", "CENTER")),
          14,
          H - 14,
          26,
          WHITE,
          "lb",
      )
    else:
      T(
          S("내 손: ", "You: ")
          + (hname(st, live_pose) if live_pose else "?"),
          14,
          H - 14,
          26,
          WHITE,
          "lb",
      )
  T(S("r: 처음부터   ESC: 종료", "r: reset   ESC: quit"), W - 14, H - 14, 22, GRAY, "rb")

  if phase == "over":
    frame[:] = (frame * 0.45).astype(np.uint8)
    win = g.over == "player"
    T(
        S("승리!", "YOU WIN!") if win else S("패배...", "YOU LOSE"),
        cx,
        cy - 120,
        110,
        PLAYER_COL if win else CPU_COL,
        "mm",
    )
    T(
        S("SPACE: 다시 시작   ESC: 종료", "SPACE: restart   ESC: quit"),
        cx,
        cy + 120,
        28,
        WHITE,
        "mm",
    )

  return flush_texts(frame)


def main():
  pipeline = (
      "v4l2src device=/dev/video0 ! "
      "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
      "videoconvert ! video/x-raw,format=BGR ! "
      "queue leaky=downstream max-size-buffers=1 "
      "max-size-bytes=0 max-size-time=0 ! "
      "appsink drop=true max-buffers=1 sync=false"
  )
  cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

  if not cap.isOpened():
    print("GStreamer 파이프라인을 열 수 없습니다.")
    return

  mp_draw = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands

  game = Game()
  phase = "idle"
  t0 = t_result = 0.0
  seen_since = None
  cpu_choice, samples, result = None, [], None
  live = deque(maxlen=5)

  # 미디어파이프 최신 버전(1.x / 0.10+) 권장 컨텍스트 관리자 적용
  with mp_hands.Hands(
      max_num_hands=1,
      model_complexity=0,
      min_detection_confidence=0.6,
      min_tracking_confidence=0.6,
  ) as hands:
    try:
      while True:
        ok, frame = cap.read()
        if not ok:
          print("프레임을 읽을 수 없습니다.")
          break

        frame = cv2.resize(cv2.flip(frame, 1), (W, H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        try:
          res = hands.process(rgb)
        except Exception as e:
          print(f"[MediaPipe process error] {e}")
          res = None

        now = time.time()
        pts = None

        if res is not None and getattr(res, "multi_hand_landmarks", None):
          lms = res.multi_hand_landmarks[0]
          try:
            pts = np.array(
                [[p.x * W, p.y * H] for p in lms.landmark],
                dtype=np.float32,
            )
          except Exception as e:
            print(f"[landmark parse error] {e}")
            pts = None

          if pts is not None and len(pts) >= 21:
            try:
              mp_draw.draw_landmarks(
                  frame,
                  lms,
                  mp_hands.HAND_CONNECTIONS,
                  mp_draw.DrawingSpec(
                      color=(255, 255, 255), thickness=1, circle_radius=2
                  ),
                  mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1),
              )
            except Exception as e:
              print(f"[draw_landmarks error] {e}")

        live.append(classify_hand(pts) if pts is not None else None)
        vals = [v for v in live if v]
        live_pose = Counter(vals).most_common(1)[0][0] if vals else None
        seen_since = (
            (seen_since if seen_since is not None else now)
            if pts is not None
            else None
        )

        if phase == "idle":
          if seen_since is not None and now - seen_since >= READY_HOLD:
            cpu_choice = pick(game.stage)
            samples, t0, phase = [], now, "count"

        elif phase == "count":
          t = now - t0
          if CAP0 <= t < CAP1:
            samples.append(sample_of(game.stage, pts))
          if t >= CAP1:
            valid = [s for s in samples if s]
            stage_now = game.stage
            if len(valid) < MIN_SAMPLES:
              result = (
                  "invalid",
                  S("인식 실패 — 다시!", "No input - again!"),
                  None,
                  cpu_choice,
                  stage_now,
              )
            else:
              p = Counter(valid).most_common(1)[0][0]
              kind, text = game.resolve(p, cpu_choice)
              result = (kind, text, p, cpu_choice, stage_now)
            phase = "over" if game.over else "result"
            t_result = now

        elif phase == "result":
          if now - t_result >= RESULT_T:
            phase, seen_since = "idle", None

        t = now - t0 if phase == "count" else 0.0
        try:
          out = draw_ui(frame, game, phase, t, result, live_pose, pts)
        except Exception as e:
          print(f"[draw_ui error] {e}")
          out = frame
        cv2.imshow("RPS -> Mukjjippa -> Chamchamcham", out)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
          break
        if k == ord("r") or (k == 32 and phase == "over"):
          game.reset()
          phase, result, seen_since = "idle", None, None

    finally:
      cap.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
