python - <<'PY'
import secrets, base64, os
print(base64.urlsafe_b64encode(secrets.token_bytes(64)).decode())
PY
