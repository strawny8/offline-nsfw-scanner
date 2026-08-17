import os
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from nudenet import NudeDetector
from tqdm import tqdm
import sys
import shutil
from datetime import datetime
import glob
import uuid
import cv2
import zipfile
import signal

try:
    import py7zr
    HAS_7Z = True
except ImportError:
    HAS_7Z = False

try:
    import rarfile
    HAS_RAR = True
except ImportError:
    HAS_RAR = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff')

all_labels = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

found = 0
default_detection_score = 0.6
previous_report_number = None
output_dir = 'reports'

report_lock = threading.Lock()
state_lock  = threading.Lock()
_shutdown   = threading.Event()

# Per-thread NudeDetector — NudeDetector is not thread-safe
_thread_local = threading.local()

def get_detector():
    if not hasattr(_thread_local, 'detector'):
        _thread_local.detector = NudeDetector()
    return _thread_local.detector

# ---------------------------------------------------------------------------
# Cache / Directory helpers
# ---------------------------------------------------------------------------

def create_cache_directory():
    os.makedirs('cache', exist_ok=True)

def get_cache_filename(filename):
    return os.path.join('cache', filename)

def create_reports_directory():
    formatted_date = datetime.now().strftime("%d%m%Y_%H%M")
    reports_dir = os.path.join(output_dir, f"reports_{formatted_date}")
    os.makedirs(reports_dir, exist_ok=True)

def get_latest_report_directory():
    pattern = os.path.join(output_dir, 'reports_*')
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    return dirs[-1] if dirs else None

def get_report_filename(report_number, local=False):
    if local:
        return f'nudenet_report_{report_number}.html'
    latest = get_latest_report_directory()
    base = latest if latest else output_dir
    return os.path.join(base, f'nudenet_report_{report_number}.html')

# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = 'scan_checkpoint.json'

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f).get('scanned', []))
        except Exception:
            pass
    return set()

def save_checkpoint(scanned_paths):
    with state_lock:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump({'scanned': list(scanned_paths)}, f)

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def create_new_report(report_number, summary=False):
    report_file = get_report_filename(report_number)
    if not os.path.exists(report_file):
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(get_report_summary() if summary else get_report_header(report_number))
    else:
        print(f"Report file already exists: {report_file}")


def update_report(report_file, image_path, matched_classes, display_path=None):
    """Inject detection card before </ul> so HTML stays valid."""
    label_path = display_path if display_path else image_path
    image_path_escaped = image_path.replace('\\', '\\\\').replace("'", "\\'")
    label_path_escaped = label_path.replace('\\', '\\\\').replace("'", "\\'")
    file_name = os.path.basename(label_path)
    matched_str = ',<br>'.join(f"{i['class']} [{i['score']:.2f}]" for i in matched_classes)
    avg_score = round(sum(i['score'] for i in matched_classes) / len(matched_classes), 2) if matched_classes else 0
    clipboard_emoji = '\U0001F4CB'

    card = f"""
        <li>
            <a href="{image_path_escaped}" target="_blank">
                <div class="zoom-container">
                    <img src='{image_path_escaped}' alt='Detected image' loading="lazy">
                </div>
            </a>
            <br>
            <div class="file-info">
            <span class="file-path-label" onclick="copyToClipboard('{label_path_escaped}')" title="{label_path_escaped}">{file_name}</span>
            <span class="file-description">Matched Classes:<br>{matched_str}<br></span>
            <span class="average-scores">[avg: {avg_score}]</span>
            <span class="clipboard-button" onclick="copyToClipboard('{label_path_escaped}')"><button>{clipboard_emoji}</button></span>
            </div>
        </li>
"""
    with report_lock:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('</ul>', card + '</ul>', 1) if '</ul>' in content else content + card
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)


def get_report_header(report_number):
    prev_link = ''
    next_link = ''
    if previous_report_number:
        prev_link = f'<a class="report-previous" href="{get_report_filename(previous_report_number, True)}">&#8592; Previous</a>&nbsp;&nbsp;|'
    if report_number > 0:
        next_link = f'&nbsp;&nbsp;<a class="report-next" href="{get_report_filename(report_number + 1, True)}">Next &#8594;</a>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NudeNet Detection Report #{report_number}</title>
    <style>
        :root {{
            --bg: #2e2c2c;
            --surface: #3d3b3b;
            --header-bg: #1a2e2e;
            --accent: #4f98a3;
            --text: #d4d0cd;
            --text-muted: #8a8785;
            --radius: 14px;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: system-ui, sans-serif;
            font-size: 14px;
            min-height: 100vh;
        }}
        .controls {{
            position: sticky;
            top: 0;
            z-index: 999;
            background: var(--header-bg);
            padding: 0.6rem 1rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        .controls label {{ display: flex; align-items: center; gap: 0.4rem; cursor: pointer; }}
        .report-navigation {{ margin-left: auto; display: flex; gap: 0.5rem; }}
        .report-navigation a {{
            color: var(--accent);
            text-decoration: none;
            padding: 0.25rem 0.6rem;
            border: 1px solid var(--accent);
            border-radius: 6px;
            font-size: 13px;
            transition: background 0.15s;
        }}
        .report-navigation a:hover {{ background: rgba(79,152,163,0.15); }}
        ul {{
            list-style: none;
            padding: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
        }}
        li {{
            background: var(--surface);
            border-radius: var(--radius);
            border: 1px solid rgba(255,255,255,0.07);
            display: flex;
            overflow: hidden;
            transition: box-shadow 0.2s;
            max-width: 460px;
        }}
        li:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
        .zoom-container {{
            width: 150px;
            height: 150px;
            flex-shrink: 0;
            overflow: hidden;
        }}
        .zoom-container img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: blur(20px);
            transition: filter 0.2s, transform 0.2s;
            cursor: pointer;
        }}
        .zoom-container img.unblurred {{ filter: none; }}
        .zoom-container img:hover {{ transform: scale(1.05); }}
        .file-info {{
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            min-width: 0;
            flex: 1;
        }}
        .file-path-label {{
            font-weight: 600;
            word-break: break-all;
            font-size: 13px;
            cursor: pointer;
            color: var(--accent);
        }}
        .file-path-label:hover {{ text-decoration: underline; }}
        .file-description {{ font-size: 12px; color: var(--text-muted); line-height: 1.5; }}
        .average-scores {{ font-size: 12px; color: var(--text-muted); }}
        .clipboard-button button {{
            background: rgba(79,152,163,0.15);
            border: 1px solid rgba(79,152,163,0.3);
            border-radius: 6px;
            padding: 0.2rem 0.5rem;
            cursor: pointer;
            color: var(--text);
            font-size: 14px;
            transition: background 0.15s;
        }}
        .clipboard-button button:hover {{ background: rgba(79,152,163,0.3); }}
        .toast {{
            position: fixed;
            bottom: 1.5rem;
            left: 50%;
            transform: translateX(-50%) translateY(20px);
            background: var(--accent);
            color: #fff;
            padding: 0.5rem 1.2rem;
            border-radius: 999px;
            font-size: 13px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s, transform 0.2s;
            z-index: 9999;
        }}
        .toast.show {{ opacity: 1; transform: translateX(-50%) translateY(0); }}
    </style>
    <script>
        function copyToClipboard(text) {{
            navigator.clipboard.writeText(text).then(() => showToast('Path copied!')).catch(() => {{
                const ta = document.createElement('textarea');
                ta.value = text; document.body.appendChild(ta); ta.select();
                document.execCommand('copy'); document.body.removeChild(ta);
                showToast('Path copied!');
            }});
        }}
        function showToast(msg) {{
            const t = document.getElementById('toast');
            t.textContent = msg; t.classList.add('show');
            setTimeout(() => t.classList.remove('show'), 2000);
        }}
        function toggleBlur() {{
            const checked = document.getElementById('blurCheckbox').checked;
            document.querySelectorAll('.zoom-container img').forEach(img => {{
                img.classList.toggle('unblurred', !checked);
            }});
        }}
    </script>
</head>
<body>
<div class="controls">
    <label><input type="checkbox" id="blurCheckbox" checked onchange="toggleBlur()"> Blur images</label>
    <span style="color:var(--text-muted);font-size:12px;">Click image to open full size &bull; Click filename to copy path</span>
    <div class="report-navigation">{prev_link}{next_link}</div>
</div>
<ul>
<!-- detections appended below -->
</ul>
<div class="toast" id="toast"></div>
</body>
</html>
"""


def get_report_summary():
    prev_link = ''
    if previous_report_number:
        prev_link = f'<a href="{get_report_filename(previous_report_number, True)}">&#8592; Previous Report</a>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scan Summary</title>
    <style>
        body {{ background:#2e2c2c; color:#d4d0cd; font-family:system-ui,sans-serif;
               display:flex; flex-direction:column; align-items:center;
               justify-content:center; min-height:100vh; gap:1rem; }}
        h1 {{ font-size:2rem; color:#4f98a3; }}
        .stat {{ font-size:1.2rem; }}
        a {{ color:#4f98a3; }}
    </style>
</head>
<body>
    <h1>Scan Complete</h1>
    <div class="stat">Total detections: <strong>{found}</strong></div>
    <div>{prev_link}</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(filename):
    _, ext = os.path.splitext(filename)
    return uuid.uuid4().hex + ext

def create_error_log_directory():
    os.makedirs(os.path.join('logs', 'error_logs'), exist_ok=True)

def get_error_log_filename(n):
    return os.path.join('logs', 'error_logs', f'error_log_{n}.txt')

def create_new_error_log(n):
    with open(get_error_log_filename(n), 'w', encoding='utf-8') as f:
        f.write("Error Log:\n")

def log_error(n, message):
    with open(get_error_log_filename(n), 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()} - {message}\n")

def clean_cache_directory(cache_dir, max_size_bytes):
    """Evict oldest flat files in cache_dir. Skips subdirs (e.g. active zip_temp_* folders)."""
    try:
        entries = [
            os.path.join(cache_dir, fn)
            for fn in os.listdir(cache_dir)
            if os.path.isfile(os.path.join(cache_dir, fn))
        ]
        current_size = sum(os.path.getsize(fp) for fp in entries)
        if current_size <= max_size_bytes:
            return
        entries.sort(key=os.path.getmtime)
        for fp in entries:
            if current_size <= max_size_bytes:
                break
            sz = os.path.getsize(fp)
            try:
                os.remove(fp)
                current_size -= sz
            except Exception:
                pass
    except Exception:
        pass

def is_complex_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    c_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.Canny(image, 100, 200)
    c_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(c_thresh) > 5 or len(c_edges) > 5

# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

def scan_image(full_path, error_log_number):
    """
    Scan a single image using a per-thread NudeDetector.
    Returns matched_classes on detection, [] on clean, None on error/skip.
    """
    sanitized = sanitize_filename(os.path.basename(full_path))
    nude_detector = get_detector()

    try:
        img = Image.open(full_path)
        img.close()
    except Exception as e:
        log_error(error_log_number, f"Cannot open {full_path}: {e}")
        temp_path = get_cache_filename(sanitized)
        try:
            shutil.copyfile(full_path, temp_path)
            full_path = temp_path
        except Exception as e2:
            log_error(error_log_number, f"Cannot copy to cache {full_path}: {e2}")
            return None

    try:
        if not is_complex_image(full_path):
            return None
        detections = nude_detector.detect(full_path)
    except Exception as e:
        log_error(error_log_number, f"Detection error {full_path}: {e}")
        temp_path = get_cache_filename(sanitized)
        try:
            shutil.copyfile(full_path, temp_path)
            if not is_complex_image(temp_path):
                return None
            detections = nude_detector.detect(temp_path)
        except Exception as e2:
            log_error(error_log_number, f"Cached detection error {full_path}: {e2}")
            return None

    matched = [
        {'class': d['class'], 'score': d['score']}
        for d in detections
        if d['class'] in all_labels and d.get('score', 0) > default_detection_score
    ]
    return matched if any(d['class'] in all_labels for d in matched) else []


def handle_detection(matched_classes, full_path, report_number_ref, images_with_detections, display_path=None):
    """Thread-safe detection writer. report_number_ref is a list[int] mutable box."""
    global found, previous_report_number

    label = display_path or full_path
    print(f"  [DETECTED] {os.path.basename(label)}", flush=True)

    with report_lock:
        found += 1
        images_with_detections.append(label)
        report_number = report_number_ref[0]
        if os.path.getsize(get_report_filename(report_number)) >= 200 * 1024:
            previous_report_number = report_number
            report_number += 1
            report_number_ref[0] = report_number
            create_new_report(report_number)
        update_report(get_report_filename(report_number), full_path, matched_classes, display_path=display_path)

# ---------------------------------------------------------------------------
# Archive scanning (.zip / .7z / .rar)
# ---------------------------------------------------------------------------

def _extract_archive(archive_path, temp_dir, error_log_number):
    """
    Extract any supported archive type to temp_dir with live progress output.
    Returns True on success, False on failure.
    """
    lower = archive_path.lower()
    name = os.path.basename(archive_path)
    try:
        if lower.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                members = zf.namelist()
                total = len(members)
                print(f"\n  [ZIP] Extracting {total} entries from: {name}", flush=True)
                for i, member in enumerate(members, 1):
                    zf.extract(member, temp_dir)
                    if i % 200 == 0 or i == total:
                        print(f"\r  [ZIP] Extracting... {i}/{total} entries", end='', flush=True)
                print(f"\r  [ZIP] Extraction complete: {total} entries from {name}     ", flush=True)
        elif lower.endswith('.7z'):
            if not HAS_7Z:
                log_error(error_log_number, f"py7zr not installed, skipping {archive_path}")
                return False
            print(f"\n  [7Z] Extracting: {name} (this may take a while)...", flush=True)
            with py7zr.SevenZipFile(archive_path, mode='r') as sz:
                all_files = sz.getnames()
                print(f"  [7Z] {len(all_files)} entries found — extracting...", flush=True)
                sz.extractall(path=temp_dir)
            print(f"  [7Z] Extraction complete: {name}", flush=True)
        elif lower.endswith('.rar'):
            if not HAS_RAR:
                log_error(error_log_number, f"rarfile not installed, skipping {archive_path}")
                return False
            with rarfile.RarFile(archive_path, 'r') as rf:
                members = rf.namelist()
                total = len(members)
                print(f"\n  [RAR] Extracting {total} entries from: {name}", flush=True)
                for i, member in enumerate(members, 1):
                    rf.extract(member, temp_dir)
                    if i % 200 == 0 or i == total:
                        print(f"\r  [RAR] Extracting... {i}/{total} entries", end='', flush=True)
                print(f"\r  [RAR] Extraction complete: {total} entries from {name}     ", flush=True)
        else:
            return False
        return True
    except Exception as e:
        log_error(error_log_number, f"Failed to extract {archive_path}: {e}")
        print(f"\n  [!] Failed to extract {name}: {e}", flush=True)
        return False


def scan_archive(archive_path, report_number_ref, error_log_number, images_with_detections, extensions):
    archive_uid = uuid.uuid4().hex
    temp_dir = os.path.join('cache', f'arc_temp_{archive_uid}')
    ext = os.path.splitext(archive_path)[1].upper()
    name = os.path.basename(archive_path)

    try:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"\n[{ext}] Found archive: {name}", flush=True)
        if not _extract_archive(archive_path, temp_dir, error_log_number):
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
    except Exception as e:
        log_error(error_log_number, f"Archive setup error {archive_path}: {e}")
        print(f"  [!] Archive setup error for {name}: {e}", flush=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Collect all image files from the extracted archive
    image_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(temp_dir)
        for f in files
        if f.lower().endswith(extensions)
    ]

    img_count = len(image_files)
    if img_count == 0:
        print(f"  [ARC] No scannable images found inside {name}", flush=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    print(f"  [ARC] Scanning {img_count} image(s) inside {name}...", flush=True)

    with tqdm(
        total=img_count,
        desc=f"  {ext} {name[:40]}",
        position=1,
        leave=False,
        dynamic_ncols=True,
        unit='img'
    ) as arc_pbar:
        for full_path in image_files:
            if _shutdown.is_set():
                break
            arc_pbar.set_postfix(f=os.path.basename(full_path)[:35])
            display_path = f"{archive_path} >> {os.path.relpath(full_path, temp_dir)}"
            matched = scan_image(full_path, error_log_number)
            if matched:
                handle_detection(matched, full_path, report_number_ref, images_with_detections, display_path=display_path)
            arc_pbar.update(1)

    print(f"  [ARC] Done scanning {name} — {img_count} image(s) processed", flush=True)
    shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

ARCHIVE_EXTENSIONS = ('.zip',)
if HAS_7Z:  _ARCHIVE_EXTS_7Z  = ('.7z',)
else:        _ARCHIVE_EXTS_7Z  = ()
if HAS_RAR: _ARCHIVE_EXTS_RAR = ('.rar',)
else:        _ARCHIVE_EXTS_RAR = ()
ALL_ARCHIVE_EXTENSIONS = ARCHIVE_EXTENSIONS + _ARCHIVE_EXTS_7Z + _ARCHIVE_EXTS_RAR


def scan_directory(directory, report_number_ref, error_log_number, extensions, exclude_dirs, resume=False, workers=1):
    images_with_detections = []

    create_error_log_directory()
    create_new_error_log(error_log_number)
    create_cache_directory()

    max_cache_bytes = 15 * 1024 * 1024
    scanned_paths = load_checkpoint() if resume else set()

    for subdir, dirs, files in os.walk(directory):
        if _shutdown.is_set():
            break

        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Announce each directory so user can see progress through the tree
        print(f"\n[DIR] {subdir}  ({len(files)} file(s))", flush=True)

        pbar = tqdm(
            total=len(files),
            desc=subdir[-60:],
            position=0,
            leave=False,
            dynamic_ncols=True,
            unit='files'
        )

        def process_file(file):
            if _shutdown.is_set():
                return
            full_path = os.path.abspath(os.path.join(subdir, file))

            if resume and full_path in scanned_paths:
                pbar.update(1)
                return

            fname_lower = file.lower()
            if fname_lower.endswith(ALL_ARCHIVE_EXTENSIONS):
                pbar.set_postfix(f=f'[ARC] {file[:35]}')
                scan_archive(full_path, report_number_ref, error_log_number, images_with_detections, extensions)
            elif fname_lower.endswith(extensions):
                matched = scan_image(full_path, error_log_number)
                if matched:
                    handle_detection(matched, full_path, report_number_ref, images_with_detections)
                pbar.set_postfix(f=file[:40])

            if resume:
                scanned_paths.add(full_path)
                save_checkpoint(scanned_paths)

            clean_cache_directory('cache', max_cache_bytes)
            pbar.update(1)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_file, f): f for f in files}
                for _ in as_completed(futures):
                    pass
        else:
            for file in files:
                process_file(file)

        pbar.close()

    create_new_report(report_number_ref[0], summary=True)
    if not _shutdown.is_set():
        clear_checkpoint()
    return images_with_detections

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global default_detection_score, output_dir

    parser = argparse.ArgumentParser(
        description='Offline NSFW image scanner using NudeNet.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py C:/Users/me/Pictures
  python script.py D:/ --minscore 0.7
  python script.py C:/ --exclude Windows --exclude "Program Files" --workers 4
  python script.py C:/ --extensions .jpg .jpeg .png --resume
  python script.py D:/ --output D:/scan_results

Archive support:
  .zip  built-in
  .7z   requires: pip install py7zr
  .rar  requires: pip install rarfile  (+ WinRAR/unrar binary in PATH)
"""
    )
    parser.add_argument('directory',    type=str,           help='Directory or drive to scan')
    parser.add_argument('--minscore',   type=float, default=0.6,
                        help='Minimum detection confidence (0-1), default: 0.6')
    parser.add_argument('--exclude',    action='append', default=[],
                        help='Directory names to skip (repeatable)')
    parser.add_argument('--extensions', nargs='+', default=list(DEFAULT_EXTENSIONS),
                        help=f'Image extensions to scan (default: {" ".join(DEFAULT_EXTENSIONS)})')
    parser.add_argument('--workers',    type=int, default=1,
                        help='Parallel scan workers (default: 1, try 2-4 for speed)')
    parser.add_argument('--resume',     action='store_true',
                        help='Resume interrupted scan from checkpoint')
    parser.add_argument('--output',     type=str, default='reports',
                        help='Directory to write reports into (default: reports/)')

    args = parser.parse_args()
    default_detection_score = args.minscore
    output_dir = args.output

    extensions = tuple(e if e.startswith('.') else f'.{e}' for e in args.extensions)

    def _sigint_handler(sig, frame):
        print("\n\n[!] Interrupted — saving checkpoint for --resume...")
        _shutdown.set()
    signal.signal(signal.SIGINT, _sigint_handler)

    os.makedirs(output_dir, exist_ok=True)
    create_reports_directory()
    report_number_ref = [1]
    create_new_report(report_number_ref[0])

    archive_support = ['.zip']
    if HAS_7Z:  archive_support.append('.7z')
    if HAS_RAR: archive_support.append('.rar')

    print(f"\nScanning : {args.directory}")
    print(f"Min score: {args.minscore}")
    print(f"Exts     : {', '.join(extensions)}")
    print(f"Archives : {', '.join(archive_support)}")
    print(f"Workers  : {args.workers}")
    print(f"Output   : {output_dir}")
    if args.exclude:
        print(f"Excluding: {', '.join(args.exclude)}")
    if args.resume:
        print(f"Resuming from checkpoint: {CHECKPOINT_FILE}")
    if not HAS_7Z:
        print("  [!] .7z support disabled — install py7zr: pip install py7zr")
    if not HAS_RAR:
        print("  [!] .rar support disabled — install rarfile: pip install rarfile")
    print()

    try:
        detections = scan_directory(
            args.directory,
            report_number_ref,
            error_log_number=1,
            extensions=extensions,
            exclude_dirs=set(args.exclude),
            resume=args.resume,
            workers=args.workers,
        )
    except Exception as e:
        print(f"\n[ERROR] Unexpected failure: {e}")
        detections = []

    status = "interrupted" if _shutdown.is_set() else "complete"
    print(f"\nScan {status}. {len(detections)} detection(s) — reports in '{output_dir}/'")
    print(f"FOUND: {found}")

if __name__ == "__main__":
    main()
    clean_cache_directory('cache', 1)
