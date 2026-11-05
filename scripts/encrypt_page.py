"""
Password-protect a map page for GitHub Pages, StatiCrypt-compatible, without Node.

Re-uses an existing StatiCrypt page as the shell (its password screen, salt and decrypt code)
and swaps in a newly encrypted map. Keeping the salt means "Remember me" on viewers' browsers
keeps working with the same passphrase.

    python scripts/encrypt_page.py --shell docs/searchmont-2024.html \
        --map output_corridor/searchmont_corridor_map.html --out docs/searchmont-2024.html

The passphrase is typed at a hidden prompt. It is checked against the shell's current
content first, so a typo can't lock the page with a passphrase nobody knows.
"""
import argparse
import getpass
import hashlib
import hmac
import json
import os
import re
import sys
from pathlib import Path

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

CONFIG_RE = re.compile(r'(staticryptConfig = )(\{"staticryptEncryptedMsgUniqueVariableName".*?\});')


def hash_password(password: str, salt: str) -> bytes:
    """StatiCrypt v3: three PBKDF2 rounds, each fed the previous round's hex string."""
    s = salt.encode()
    h = hashlib.pbkdf2_hmac("sha1", password.encode(), s, 1000, 32).hex()
    h = hashlib.pbkdf2_hmac("sha256", h.encode(), s, 14000, 32).hex()
    return hashlib.pbkdf2_hmac("sha256", h.encode(), s, 585000, 32)


def encode(msg: str, key: bytes) -> str:
    iv = os.urandom(16)
    p = padding.PKCS7(128).padder()
    data = p.update(msg.encode("utf-8")) + p.finalize()
    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    encrypted = (iv + enc.update(data) + enc.finalize()).hex()
    return hmac.new(key, encrypted.encode(), hashlib.sha256).hexdigest() + encrypted


def decode(payload: str, key: bytes) -> str | None:
    sig, encrypted = payload[:64], payload[64:]
    if not hmac.compare_digest(sig, hmac.new(key, encrypted.encode(), hashlib.sha256).hexdigest()):
        return None
    raw = bytes.fromhex(encrypted)
    dec = Cipher(algorithms.AES(key), modes.CBC(raw[:16])).decryptor()
    u = padding.PKCS7(128).unpadder()
    return (u.update(dec.update(raw[16:]) + dec.finalize()) + u.finalize()).decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", required=True, type=Path, help="existing StatiCrypt page to reuse")
    ap.add_argument("--map", required=True, type=Path, help="plain map HTML to encrypt")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    shell = a.shell.read_text(encoding="utf-8")
    m = CONFIG_RE.search(shell)
    if not m:
        sys.exit("couldn't find staticryptConfig in the shell page")
    cfg = json.loads(m.group(2))
    salt = cfg["staticryptSaltUniqueVariableName"]

    key = hash_password(getpass.getpass("Passphrase (hidden): "), salt)
    if decode(cfg["staticryptEncryptedMsgUniqueVariableName"], key) is None:
        sys.exit("That passphrase doesn't open the current page. Nothing was changed.")
    print("Passphrase matches the current page.")

    page = a.map.read_text(encoding="utf-8")
    cfg["staticryptEncryptedMsgUniqueVariableName"] = encode(page, key)
    if decode(cfg["staticryptEncryptedMsgUniqueVariableName"], key) != page:
        sys.exit("round-trip check failed; nothing written")
    new_cfg = json.dumps(cfg, separators=(",", ":"))
    out = shell[:m.start(2)] + new_cfg + shell[m.end(2):]
    a.out.write_text(out, encoding="utf-8", newline="")
    print(f"wrote {a.out} ({len(out) / 1e6:.1f} MB), encrypted {a.map.name}, same salt and passphrase")


if __name__ == "__main__":
    main()
