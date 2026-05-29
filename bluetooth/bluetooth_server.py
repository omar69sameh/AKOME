import argparse
import json
import os
import socket
import time

import bluetooth

IDS_FILE = os.path.join(os.path.dirname(__file__), "ids.json")

# Address the C# WinForms app connects to (same host/port in SimpleSocketClient.cs).
SOCKET_HOST = "127.0.0.1"
SOCKET_PORT = 5000

# Shorter scan timing for faster UI response.
SCAN_WITH_NAMES_SEC = 6
SCAN_ADDRESS_ONLY_SEC = 3
SLEEP_BETWEEN_SCANS_SEC = 0.5

NAME_ATTEMPTS = 4
NAME_TIMEOUT_SEC = 8
SLEEP_BETWEEN_NAME_TRIES_SEC = 0.5


def format_mac(raw):
    text = str(raw).strip().upper().replace("-", ":")
    return text


def _row(mac, device_name):
    mac = format_mac(mac)
    name = (device_name or "").strip() or mac
    return {
        "id": mac,
        "mac_address": mac,
        "device_name": name,
    }


def load_ids():
    if not os.path.isfile(IDS_FILE):
        return []
    try:
        with open(IDS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, dict):
        return []

    rows = data.get("ids")
    if not isinstance(rows, list):
        return []

    out = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        mac = format_mac(item.get("mac_address") or item.get("id") or "")
        if not mac:
            continue
        name = item.get("device_name")
        if name is None or str(name).strip() == "":
            name = mac
        out.append(_row(mac, str(name)))
    return out


def save_ids(rows):
    payload = {"ids": rows}
    with open(IDS_FILE, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def registered_macs(rows):
    return {format_mac(r["mac_address"]) for r in rows}


def scan_devices():
    seen = {}

    plan = [
        (SCAN_WITH_NAMES_SEC, True),
        (SCAN_WITH_NAMES_SEC, True),
        (SCAN_ADDRESS_ONLY_SEC, False),
    ]

    for index, (seconds, want_names) in enumerate(plan):
        raw_list = bluetooth.discover_devices(
            duration=seconds,
            lookup_names=want_names,
            flush_cache=True,
        )

        for entry in raw_list:
            if isinstance(entry, tuple) and len(entry) >= 2:
                mac, name = entry[0], entry[1]
            else:
                mac, name = entry, None

            mac = format_mac(mac)
            if not mac:
                continue

            name = (name or "").strip() or None

            if mac not in seen:
                seen[mac] = name
            elif seen[mac] is None and name is not None:
                seen[mac] = name

        if index < len(plan) - 1:
            time.sleep(SLEEP_BETWEEN_SCANS_SEC)

    devices = []

    for mac in sorted(seen.keys()):
        name = seen[mac]
        if not name:
            for _ in range(NAME_ATTEMPTS):
                try:
                    reply = bluetooth.lookup_name(mac, timeout=NAME_TIMEOUT_SEC)
                except (OSError, SystemError, TypeError, ValueError):
                    reply = None
                if reply and str(reply).strip():
                    name = str(reply).strip()
                    break
                time.sleep(SLEEP_BETWEEN_NAME_TRIES_SEC)
        if not name:
            name = mac
        devices.append({"addr": mac, "name": name})

    devices.sort(key=lambda item: (item["name"].lower(), item["addr"]))
    return devices


# TCP API only: LIST:IN then SIGNIN used to call scan_devices() twice (~1+ min total) and froze the C# UI.
# Reuse one scan for a few minutes so sign-in stays fast while the user picks a number.
_last_bt_scan_result = None
_last_bt_scan_time = 0.0
BT_SCAN_REUSE_SECONDS = 180.0


def scan_devices_for_socket():
    """Same as scan_devices() but returns cached list if it is still fresh (socket server only)."""
    global _last_bt_scan_result, _last_bt_scan_time
    now = time.time()
    if _last_bt_scan_result is not None and (now - _last_bt_scan_time) < BT_SCAN_REUSE_SECONDS:
        return _last_bt_scan_result
    _last_bt_scan_result = scan_devices()
    _last_bt_scan_time = now
    return _last_bt_scan_result


def print_device_list(devices):
    if len(devices) == 0:
        print("No devices.")
        return
    for i in range(len(devices)):
        row = devices[i]
        n = i + 1
        print(f"{n}. {row['name']}  {row['addr']}")


def pick_one_device(devices):
    print_device_list(devices)
    if len(devices) == 0:
        return None

    while True:
        typed = input(f"Pick 1–{len(devices)}, or 0 to cancel: ").strip()
        if not typed.isdigit():
            print("Enter a number.")
            continue
        choice = int(typed)
        if choice == 0:
            return None
        if 1 <= choice <= len(devices):
            return devices[choice - 1]
        print("Out of range.")


# --- Shared logic for menu and TCP server ---


def signup_save(mac, device_name):
    """
    Insert or update one device in ids.json (same rules as the old do_sign_up body).
    """
    rows = load_ids()
    mac = format_mac(mac)
    if not mac:
        return False
    label = (device_name or "").strip() or mac
    new_row = _row(mac, label)
    macs_known = registered_macs(rows)

    if mac in macs_known:
        updated = []
        for r in rows:
            if format_mac(r["mac_address"]) == mac:
                updated.append(new_row)
            else:
                updated.append(r)
        rows = updated
    else:
        rows = rows + [new_row]

    save_ids(rows)
    return True


def handle_signup_mac(mac_str):
    """
    SIGNUP:<mac> — save MAC. Friendly name from Bluetooth lookup if possible, else MAC.
    """
    mac = format_mac(mac_str)
    if not mac:
        return False
    name = mac
    try:
        reply = bluetooth.lookup_name(mac, timeout=NAME_TIMEOUT_SEC)
        if reply and str(reply).strip():
            name = str(reply).strip()
    except (OSError, Exception):
        # PyBluez may raise different types depending on version / adapter.
        pass
    return signup_save(mac, name)


def handle_signin_mac(mac_str):
    """
    SIGNIN:<mac> — SUCCESS if MAC is registered in ids.json.
    No live scan required: the device was verified at sign-up time and
    Bluetooth devices are not discoverable by default, which caused empty
    sign-in lists when the user wasn't actively in their Bluetooth settings.
    """
    mac = format_mac(mac_str)
    if not mac:
        return False
    rows = load_ids()
    reg = registered_macs(rows)
    return mac in reg


def devices_to_protocol_lines(devices):
    """Build N|MAC|Name lines (pipe in name is replaced so parsing stays simple)."""
    lines = []
    for i, d in enumerate(devices):
        n = i + 1
        mac = d["addr"]
        name = str(d["name"]).replace("|", " ")
        lines.append(f"{n}|{mac}|{name}")
    return lines


def handle_list_up():
    """LIST:UP — all devices found by scan (sign-up picker)."""
    return devices_to_protocol_lines(scan_devices_for_socket())


def handle_list_in():
    """LIST:IN — all registered devices from ids.json.
    Previously this filtered to only devices visible in a live scan, but most
    Bluetooth devices are not discoverable unless the user is actively inside
    their Bluetooth settings, which caused the sign-in list to always be empty.
    Registration at sign-up time is sufficient proof of device ownership."""
    rows = load_ids()
    if len(rows) == 0:
        return []
    # Reverse so the most recently registered device appears at the top.
    devices = [{"addr": r["mac_address"], "name": r["device_name"]} for r in reversed(rows)]
    return devices_to_protocol_lines(devices)


def process_request_line(line):
    """
    One request line from C# → full response string (may be several lines, always ends with newline).
    """
    line = (line or "").strip()
    if line == "LIST:UP":
        body_lines = handle_list_up()
        text = "\n".join(body_lines) + "\nEND\n"
        return text
    if line == "LIST:IN":
        body_lines = handle_list_in()
        text = "\n".join(body_lines) + "\nEND\n"
        return text
    if line.startswith("SIGNUP:"):
        mac_part = line[7:].strip()
        ok = handle_signup_mac(mac_part)
        return "SUCCESS\n" if ok else "FAIL\n"
    if line.startswith("SIGNIN:"):
        mac_part = line[7:].strip()
        ok = handle_signin_mac(mac_part)
        return "SUCCESS\n" if ok else "FAIL\n"
    return "FAIL\n"


def run_server():
    """Wait on TCP; handle one request per connection (student-simple, no threads)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((SOCKET_HOST, SOCKET_PORT))
    srv.listen(1)
    print(f"Bluetooth socket server on {SOCKET_HOST}:{SOCKET_PORT} (Ctrl+C to stop)")
    try:
        while True:
            conn, addr = srv.accept()
            reader = None
            writer = None
            try:
                reader = conn.makefile("r", encoding="utf-8", newline="\n")
                writer = conn.makefile("w", encoding="utf-8", newline="\n")
                request_line = reader.readline()
                if request_line:
                    response = process_request_line(request_line)
                    try:
                        # If C# closes the socket first (app closed, tab switched, etc.),
                        # Windows raises ConnectionResetError here — not a bug in your logic.
                        writer.write(response)
                        writer.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print(
                            "Note: client disconnected before the reply finished "
                            "(server keeps running)."
                        )
            finally:
                for f in (reader, writer):
                    if f is not None:
                        try:
                            f.close()
                        except OSError:
                            pass
                conn.close()
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        srv.close()


def do_sign_up():
    rows = load_ids()
    devices = scan_devices()
    chosen = pick_one_device(devices)
    if chosen is None:
        return

    mac = format_mac(chosen["addr"])
    label = chosen["name"]
    if signup_save(mac, label):
        print("OK")


def do_sign_in():
    rows = load_ids()
    if len(rows) == 0:
        print("No registered devices.")
        return

    devices = [{"addr": r["mac_address"], "name": r["device_name"]} for r in rows]

    if len(devices) == 1:
        print(f"Signed in as: {devices[0]['name']}  {devices[0]['addr']}")
        return

    picked = pick_one_device(devices)
    if picked is None:
        return
    print(f"Signed in as: {picked['name']}  {picked['addr']}")


def main():
    while True:
        print("1=sign up 2=sign in 0=exit")
        choice = input("Choice: ").strip()

        if choice == "1":
            do_sign_up()
        elif choice == "2":
            do_sign_in()
        elif choice == "0":
            break
        else:
            print("Bad choice.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bluetooth ids.json helper — menu or TCP server for TUIO demo."
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Listen on localhost for C# (LIST:UP, LIST:IN, SIGNUP:, SIGNIN:).",
    )
    args = parser.parse_args()
    if args.server:
        run_server()
    else:
        main()
