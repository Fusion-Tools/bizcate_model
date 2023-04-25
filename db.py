# %%
import os
from dotenv import load_dotenv
from fusion_db import FusionDB
from siuba import *

# %%
load_dotenv()
DB_USER = os.getenv("db_username")
DB_PASSWORD = os.getenv("db_password")

# %%
fdb = FusionDB(user=DB_USER, password=DB_PASSWORD, role=None)

# %%
# fmt: off
month_mapping = (
    fdb.LOOKUP.ZZINFO.F005_MONTH(lazy=True) 
    >> select("MONTH_NUM", "MONTH_YEAR")
)
# fmt: on
