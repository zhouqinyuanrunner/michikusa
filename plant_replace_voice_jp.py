import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import math, random
from functools import lru_cache
import threading, queue, time, re, os, sys

# ===================== 开关与路径 =====================
USE_VOSK_ASR = True           # True：用 Vosk 离线识别；False：不启用麦克风（配合 TEST_MODE 可模拟一条消息）
TEST_MODE = False             # 仅在 USE_VOSK_ASR=False 时有效：True=模拟一次 “こんにちは”

# —— Vosk（日语优先，找不到则回退中文）——
VOSK_MODEL_PATH_JA = os.path.expanduser("~/vosk-models/vosk-model-small-ja-0.22")
VOSK_MODEL_PATH_ZH = os.path.expanduser("~/vosk-models/vosk-model-small-cn-0.22")

# ============ 可调参数 ============
TARGET_WIDTH = 960
CONF_THRESH = 0.4
IOU_THRESH = 0.5
MASK_THRESHOLD = 0.5

# 覆盖参数
PLANT_OPACITY = 1.0
BLUR_KSIZE = 25
LEAF_IMAGE_PATH = "/Users/zhouqinyuan/Desktop/plant/leave.png"  # 使用 PNG

# 语音参数（仅在 USE_VOSK_ASR=True 有效）
MIC_DEVICE = None
ASR_SR = 16000
ASR_BLOCKSIZE = 8000

# —— Chat 窗口参数（更宽 + 像素级自动换行 + 固定第一句）——
CHAT_WIN_WIDTH = 900
CHAT_WIN_HEIGHT = 480
CHAT_MAX_LINES = 18
CHAT_TITLE = "チャット端末"
PIN_FONT_SIZE = 16
PIN_BG = True

# —— 摄像头叠加对话样式（小而轻，不挡画面）——
OVERLAY_FONT_SIZE = 14
OVERLAY_LINE_H = 20
OVERLAY_PADDING = 6
OVERLAY_BG_ALPHA = 0.22
OVERLAY_MAX_CHARS = 60

# —— 字体路径（用于中/日文渲染）——
CANDIDATE_FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB W3.ttc",
    "/Library/Fonts/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]

# —— 固定第一句（只在 Chat 窗口常驻）——
HEADER_MSG = "初めまして、私はみんなを道草に変える写真装置です。気軽に話しかけてくださいね。"

# ===================== 文本与绘制 =====================
def get_font(font_size=18):
    for p in CANDIDATE_FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                continue
    return ImageFont.load_default()

def put_text_cn(img_bgr, text, org, font_size=18, color=(240,240,240), anchor="lt"):
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size)
    x, y = org
    if anchor == "lb":
        ascent, descent = font.getmetrics()
        y = y - (ascent + descent)
    r, g, b = int(color[2]), int(color[1]), int(color[0])
    draw.text((x, y), text, font=font, fill=(r, g, b, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def wrap_text_to_width(txt, font, max_px):
    """按像素宽度换行（Pillow>=8 有 getlength；若无则按字符估算）"""
    if not txt:
        return [""]
    lines = []
    cur = ""
    for seg in str(txt).split("\n"):
        if not seg:
            lines.append("")
            continue
        for ch in seg:
            test = cur + ch
            try:
                w = font.getlength(test)  # Pillow 8+
            except Exception:
                fs = font.size if hasattr(font, "size") else 18
                w = sum((0.9*fs if ord(c)>255 else 0.6*fs) for c in test)
            if w <= max_px:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                    cur = ch
                else:
                    lines.append(ch)
                    cur = ""
        if cur:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines

def draw_chat_window(chat_history, width=CHAT_WIN_WIDTH, height=CHAT_WIN_HEIGHT, title=CHAT_TITLE):
    pad = 14
    line_h = 24
    bg = np.full((height, width, 3), 18, dtype=np.uint8)

    # 标题栏
    cv2.rectangle(bg, (0, 0), (width, 36), (40, 40, 40), -1)
    bg = put_text_cn(bg, title, (10, 8), font_size=18, color=(220,220,220), anchor="lt")

    # 固定第一句（常驻）
    y = 46
    if PIN_BG:
        cv2.rectangle(bg, (0, y-2), (width, y+PIN_FONT_SIZE+12), (28, 28, 28), -1)
    bg = put_text_cn(bg, HEADER_MSG, (pad, y), font_size=PIN_FONT_SIZE, color=(240,240,240), anchor="lt")
    y += PIN_FONT_SIZE + 16

    # 分隔线
    cv2.line(bg, (0, y), (width, y), (60, 60, 60), 1)
    y += 10

    # 对话正文（YOU/パソコン），像素级换行
    content_w = width - pad*2
    font = get_font(18)
    expanded = []
    for who, msg in chat_history[-120:]:
        prefix = "あなた> " if who == "YOU" else "パソコン> "
        expanded.extend(wrap_text_to_width(prefix + msg, font, content_w))

    expanded = expanded[-CHAT_MAX_LINES:]  # 只显示最后 N 行
    for line in expanded:
        if y > height - pad:
            break
        bg = put_text_cn(bg, line, (pad, y), font_size=18, color=(240,240,240), anchor="lt")
        y += line_h

    return bg

# ===================== 摄像头打开（稳健） =====================
def try_open_camera(index, backend=None, width=TARGET_WIDTH, warmup_frames=3):
    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    for _ in range(warmup_frames):
        ok, _ = cap.read()
        if not ok:
            time.sleep(0.05)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def open_camera_robust():
    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(None)

    tried = []
    for be in backends:
        for idx in range(0, 4):
            cap = try_open_camera(idx, backend=be)
            tried.append((idx, be))
            if cap is not None:
                be_name = "AVFoundation" if be == cv2.CAP_AVFOUNDATION else "Default"
                print(f"[Camera] 打开成功：index={idx}, backend={be_name}")
                return cap
    print("[Camera] 尝试失败，已尝试：", tried)
    raise RuntimeError("无法打开摄像头。请检查权限与设备连接。")

# ===================== 贴图与合成 =====================
def load_image_bgra(path: str) -> np.ndarray:
    """读取任意图片并转为 BGRA；若无 alpha 则补全不透明 alpha。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"未找到贴图：{path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.full(b.shape, 255, dtype=np.uint8)
        img = cv2.merge([b, g, r, a])
    elif img.shape[2] == 4:
        pass
    else:
        raise ValueError("不支持的贴图通道数")
    return img

def resize_bgra_keep_aspect(img_bgra: np.ndarray, target_w: int, max_h: int) -> np.ndarray:
    """按目标宽度等比缩放，且高度不超过 max_h。"""
    h, w = img_bgra.shape[:2]
    if w <= 0 or h <= 0:
        return img_bgra
    scale = target_w / float(w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_h > max_h:
        scale2 = max_h / float(new_h)
        new_w = max(1, int(round(new_w * scale2)))
        new_h = max(1, int(round(new_h * scale2)))
    return cv2.resize(img_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

def overlay_bgra(bg_bgr, fg_bgra, x, y, opacity=1.0):
    """将 fg_bgra 以左上角 (x,y) 叠到 bg_bgr，自动裁剪越界"""
    h, w = bg_bgr.shape[:2]
    fh, fw = fg_bgra.shape[:2]
    if fw <= 0 or fh <= 0:
        return
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + fw), min(h, y + fh)
    if x1 >= x2 or y1 >= y2:
        return
    fg_x1, fg_y1 = x1 - x, y1 - y
    fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)
    roi_bg = bg_bgr[y1:y2, x1:x2]
    roi_fg = fg_bgra[fg_y1:fg_y2, fg_x1:fg_x2]
    if roi_fg.shape[:2] != roi_bg.shape[:2]:
        roi_fg = cv2.resize(roi_fg, (roi_bg.shape[1], roi_bg.shape[0]), interpolation=cv2.INTER_LINEAR)
    fg_rgb = roi_fg[:, :, :3].astype(np.float32)
    fg_alpha = (roi_fg[:, :, 3:4].astype(np.float32) / 255.0) * float(opacity)
    bg_rgb = roi_bg.astype(np.float32)
    out = fg_rgb * fg_alpha + bg_rgb * (1.0 - fg_alpha)
    roi_bg[:] = np.clip(out, 0, 255).astype(np.uint8)

def blur_region_by_mask(frame_bgr, mask_uint8, ksize=35):
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(frame_bgr, (ksize, ksize), 0)
    mask3 = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
    inv = cv2.bitwise_not(mask3)
    fg = cv2.bitwise_and(frame_bgr, inv)
    bg = cv2.bitwise_and(blurred, mask3)
    return cv2.add(fg, bg)

def blur_upper_person(frame_bgr, full_mask_uint8, y_top, y_bottom, frac=0.6, ksize=25):
    """只模糊人框上半部（大概脸）"""
    mask = full_mask_uint8.copy()
    H = mask.shape[0]
    y1 = max(0, min(H-1, int(y_top)))
    y2 = max(0, min(H,   int(y_bottom)))
    cut = int(y1 + (y2 - y1) * frac)
    if cut <= y1:
        return frame_bgr
    mask[y2:,:] = 0
    mask[cut:,:] = 0
    return blur_region_by_mask(frame_bgr, mask, ksize=ksize)

# ===================== 程序化植物（兜底） =====================
@lru_cache(maxsize=32)
def generate_plant_png_cached(width, height, seed=0):
    random.seed(seed)
    return generate_plant_png(width, height, seed)

def generate_plant_png(width, height, seed=0):
    W = max(32, int(width))
    H = max(32, int(height))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base_x = W // 2
    base_y = H - 5
    def draw_branch(x, y, length, angle, thickness, depth):
        if depth <= 0 or length < 6:
            leaf_size = max(3, int(thickness * 1.5))
            color_leaf = (
                int(60 + random.random()*120),
                int(120 + random.random()*120),
                int(60 + random.random()*100),
                200 + int(random.random()*55)
            )
            draw.ellipse([x - leaf_size, y - leaf_size, x + leaf_size, y + leaf_size], fill=color_leaf)
            return
        rad = math.radians(angle)
        x2 = x + length * math.cos(rad)
        y2 = y - length * math.sin(rad)
        color_trunk = (int(90 + random.random()*40), int(50 + random.random()*30), 10, 255)
        draw.line([x, y, x2, y2], fill=color_trunk, width=max(1, int(thickness)))
        branch_num = 2 if random.random() < 0.85 else 3
        for i in range(branch_num):
            new_len = length * (0.65 + random.random()*0.15)
            new_ang = angle + (random.uniform(-28, -12) if i % 2 == 0 else random.uniform(12, 28))
            new_thick = max(1.0, thickness * (0.7 + random.random()*0.1))
            draw_branch(x2, y2, new_len, new_ang, new_thick, depth - 1)
    trunk_len = H * (0.45 + random.random()*0.1)
    draw_branch(base_x, base_y, trunk_len, 90 + random.uniform(-3, 3), thickness=max(2, W*0.02), depth=5)
    return img

# ===================== 本地 Chat 规则（默认日语） =====================
def generate_reply_local(text: str) -> str:
    t = text.strip().lower()
    rules = [
        (r"(你好|您好|嗨|hey|hello|hi)", "こんにちは。"),
        (r"(こんばんは|おはよう|こんにちは)", "こんにちは。"),
        (r"(你是谁|你是誰|你是什么)", "私はあなたのビジュアル・アシスタントである。"),
        (r"(再见|拜拜|bye)", "では、また。"),
        (r"(谢谢|ありがと|感謝)", "どういたしまして。"),
    ]
    for p, resp in rules:
        if re.search(p, t):
            return resp
    return "了解した。"

# ===================== （可选）Vosk 离线 ASR（JA優先、ZHフォールバック） =====================
def pick_vosk_model_path():
    if os.path.isdir(VOSK_MODEL_PATH_JA):
        return VOSK_MODEL_PATH_JA
    if os.path.isdir(VOSK_MODEL_PATH_ZH):
        print(f"[VOSK] 日本語モデル未検出、中文モデルへフォールバック：{VOSK_MODEL_PATH_ZH}")
        return VOSK_MODEL_PATH_ZH
    return None

def list_audio_devices():
    try:
        import sounddevice as sd
        print("\n利用可能なオーディオデバイス（'max_input_channels' > 0 が録音可）：")
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            mark = "IN " if dev["max_input_channels"] > 0 else "   "
            print(f"[{idx:2d}] {mark}{dev['name']}  (inputs={dev['max_input_channels']}, outputs={dev['max_output_channels']})")
        print(f"\n現在の MIC_DEVICE = {MIC_DEVICE}（None はデフォルト）\n")
    except Exception as e:
        print("デバイス一覧の取得に失敗：", e)

def asr_worker_vosk(text_queue: queue.Queue, stop_event: threading.Event):
    try:
        import sounddevice as sd
        from vosk import Model, KaldiRecognizer
    except Exception as e:
        print("Vosk もしくは sounddevice の読み込みに失敗：", e); return

    list_audio_devices()
    model_path = pick_vosk_model_path()
    if not model_path:
        print("Vosk モデルが見つかりません。日本語モデルを推奨：vosk-model-small-ja-0.22")
        return

    try:
        model = Model(model_path)
        rec = KaldiRecognizer(model, ASR_SR)
        rec.SetWords(True)
        print(f"Vosk モデルを読み込み：{model_path}")
    except Exception as e:
        print(f"Vosk モデルロード失敗：{e}")
        return

    audio_q = queue.Queue(maxsize=20)
    def audio_callback(indata, frames, time_info, status):
        if status: pass
        try: audio_q.put(bytes(indata))
        except queue.Full: pass

    try:
        with sd.RawInputStream(
            samplerate=ASR_SR, blocksize=ASR_BLOCKSIZE, dtype="int16", channels=1,
            callback=audio_callback, device=MIC_DEVICE
        ):
            print("マイクを起動（Vosk）。")
            partial_cache = ""
            last_emit_ts = time.time()
            while not stop_event.is_set():
                try:
                    data = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    res = rec.Result()
                    m = re.search(r'"text"\s*:\s*"([^"]*)"', res)
                    if m:
                        txt = m.group(1).strip()
                        if txt:
                            text_queue.put(txt)
                            partial_cache = ""
                            last_emit_ts = time.time()
                else:
                    pres = rec.PartialResult()
                    m = re.search(r'"partial"\s*:\s*"([^"]*)"', pres)
                    if m:
                        partial = m.group(1).strip()
                        if partial and (partial != partial_cache) and (time.time() - last_emit_ts > 2.0):
                            text_queue.put(partial)
                            partial_cache = partial
                            last_emit_ts = time.time()
    except Exception as e:
        print(f"音声キャプチャ例外：{e}")

# ===================== 主流程 =====================
def main():
    model = YOLO("yolov8n-seg.pt")
    names = model.model.names if hasattr(model.model, 'names') else {}

    try:
        leaf_bgra_orig = load_image_bgra(LEAF_IMAGE_PATH)
    except Exception as e:
        print(f"贴图読み込み失敗：{e}")
        leaf_bgra_orig = None

    cap = open_camera_robust()
    CAM_WIN = "Plant Replace (q/Esc退出, s保存)"
    CHAT_WIN = "Chat Terminal"

    # 摄像头窗口：全屏显示
    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(CAM_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass

    print("q / Esc 退出，s 保存 snapshot.jpg。")

    text_queue = queue.Queue()
    stop_event = threading.Event()

    chat_history = []
    last_user_text, last_bot_text = "", ""
    overlay_fade_count = 0     # >0 时显示摄像头对话叠加

    if USE_VOSK_ASR:
        asr_thread = threading.Thread(target=asr_worker_vosk, args=(text_queue, stop_event), daemon=True)
        asr_thread.start()
    else:
        if TEST_MODE:
            text_queue.put("こんにちは")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读帧失败，尝试重新打开摄像头……")
                cap.release()
                time.sleep(0.3)
                try:
                    cap = open_camera_robust()
                except Exception as e:
                    print(f"重连摄像头失败：{e}")
                    break
                continue

            h, w = frame.shape[:2]
            scale = TARGET_WIDTH / float(w)
            if abs(scale - 1.0) > 1e-3:
                frame = cv2.resize(frame, (TARGET_WIDTH, int(h*scale)), interpolation=cv2.INTER_LINEAR)
            H, W = frame.shape[:2]

            # —— YOLO 分割：找人并覆盖 ——
            results = model.predict(
                source=frame,
                verbose=False,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                imgsz=640,
                half=False,
                device="cpu"
            )

            if len(results):
                r = results[0]
                if r.boxes is not None and r.masks is not None:
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(int).tolist()
                    masks = r.masks.data.cpu().numpy()
                    for i, cid in enumerate(cls_ids):
                        is_person = (cid == 0) or (names.get(cid, "").lower() == "person")
                        if not is_person:
                            continue
                        mask = (masks[i] > MASK_THRESHOLD).astype(np.uint8) * 255
                        if mask.shape[:2] != (H, W):
                            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        x1, y1, x2, y2 = xyxy[i]
                        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
                        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
                        bw, bh = x2 - x1, y2 - y1
                        if bw * bh < 800:
                            continue

                        frame = blur_upper_person(frame, mask, y1, y2, frac=0.6, ksize=BLUR_KSIZE)

                        target_w = max(30, int(bw * 0.8))
                        max_h = max(40, int(bh * 0.6))
                        if leaf_bgra_orig is not None:
                            leaf_resized = resize_bgra_keep_aspect(leaf_bgra_orig, target_w, max_h)
                        else:
                            pil_img = generate_plant_png_cached(target_w, max_h, seed=target_w*1000 + max_h)
                            leaf_resized = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

                        lh, lw = leaf_resized.shape[:2]
                        cx = x1 + bw // 2
                        cy = int(y1 + bh * 0.25)
                        px = int(cx - lw / 2)
                        py = int(cy - lh / 2)
                        overlay_bgra(frame, leaf_resized, px, py, opacity=PLANT_OPACITY)

            # —— 处理语音文本（本地规则：默认日语）——
            while True:
                try:
                    user_text = text_queue.get_nowait()
                except queue.Empty:
                    break
                user_text = (user_text or "").strip()
                if not user_text:
                    continue
                chat_history.append(("YOU", user_text))
                bot_text = generate_reply_local(user_text)
                chat_history.append(("BOT", bot_text))
                last_user_text, last_bot_text = user_text, bot_text
                overlay_fade_count = 90  # ~3 秒（30fps）

            # —— 摄像头画面两行对话叠加（小号、轻透明；不说话自动消失） ——
            if overlay_fade_count > 0 and (last_user_text or last_bot_text):
                usr = (last_user_text or "")[:OVERLAY_MAX_CHARS]
                bot = (last_bot_text or "")[:OVERLAY_MAX_CHARS]

                box_w = min(460, W - 20)
                box_h = OVERLAY_PADDING*2 + OVERLAY_LINE_H*2
                top_y = 10

                overlay = frame.copy()
                cv2.rectangle(overlay, (10, top_y), (10+box_w, top_y+box_h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, OVERLAY_BG_ALPHA, frame, 1.0 - OVERLAY_BG_ALPHA, 0.0)

                tx = 10 + OVERLAY_PADDING
                ty1 = top_y + OVERLAY_PADDING
                ty2 = ty1 + OVERLAY_LINE_H

                if usr:
                    frame = put_text_cn(frame, f"あなた: {usr}", (tx, ty1), font_size=OVERLAY_FONT_SIZE, color=(255,255,255))
                if bot:
                    frame = put_text_cn(frame, f"パソコン: {bot}", (tx, ty2), font_size=OVERLAY_FONT_SIZE, color=(220,255,220))
                overlay_fade_count -= 1

            # —— 显示窗口 ——
            cv2.imshow(CAM_WIN, frame)
            chat_img = draw_chat_window(chat_history, width=CHAT_WIN_WIDTH, height=CHAT_WIN_HEIGHT, title=CHAT_TITLE)
            cv2.imshow(CHAT_WIN, chat_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                cv2.imwrite("snapshot.jpg", frame)
                print("snapshot.jpg を保存した。")

    finally:
        stop_event.set()
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

