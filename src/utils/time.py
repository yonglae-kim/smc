from __future__ import annotations
import datetime

def today_kst() -> datetime.date:
    # KST is UTC+9; keep it simple without pytz dependency.
    utc = datetime.datetime.utcnow()
    kst = utc + datetime.timedelta(hours=9)
    return kst.date()

def now_kst_iso() -> str:
    utc = datetime.datetime.utcnow()
    kst = utc + datetime.timedelta(hours=9)
    return kst.strftime("%Y-%m-%d %H:%M:%S")
