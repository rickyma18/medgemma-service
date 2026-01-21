# python - << 'PY'
import os
import firebase_admin
from firebase_admin import auth, credentials

proj = os.getenv("FIREBASE_PROJECT_ID")
sa = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

print("FIREBASE_PROJECT_ID =", proj)
print("GOOGLE_APPLICATION_CREDENTIALS =", sa)

if not firebase_admin._apps:
    if sa:
        cred = credentials.Certificate(sa)
        firebase_admin.initialize_app(cred, {"projectId": proj})
        print("init: service_account_file")
    else:
        firebase_admin.initialize_app(options={"projectId": proj})
        print("init: adc")

token = os.environ.get("TOKEN","")
try:
    decoded = auth.verify_id_token(token, check_revoked=False)
    print("verify: OK")
    print("keys:", list(decoded.keys()))
except Exception as e:
    print("verify: FAIL")
    print("error_class:", e.__class__.__name__)
    print("error_str:", str(e)[:200])

