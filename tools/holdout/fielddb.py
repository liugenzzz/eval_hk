"""Connection settings for the field MySQL DB, kept out of the repo.

Reads $FIELD_DB_CONF, default /mnt/si003010kcx0/mmdata/mm_env/field_tools/db.json:
  {"host": "...", "port": 3306, "user": "...", "password": "...", "database": "field"}
"""
import json
import os

import pymysql

CONF = os.environ.get("FIELD_DB_CONF", "/mnt/si003010kcx0/mmdata/mm_env/field_tools/db.json")


def connect(**overrides):
    with open(CONF, encoding="utf-8") as f:
        conf = json.load(f)
    conf.setdefault("database", "field")
    conf.setdefault("charset", "utf8mb4")
    conf.setdefault("autocommit", False)
    conf.update(overrides)
    return pymysql.connect(**conf)
