"""Run DDL on a shared table without blocking anyone else.

A *pending* EXCLUSIVE metadata lock queues every other query on the table behind
it, so an ALTER that camps on the lock can stall a whole team. This polls until
nobody else holds an MDL on the table, then fires the ALTER with a short
lock_wait_timeout as a backstop, and retries if it loses the race.
"""
import time

import pymysql

HOLDERS = """SELECT t.processlist_id, t.processlist_command, t.processlist_time,
                    LEFT(COALESCE(t.processlist_info, ''), 60)
             FROM performance_schema.metadata_locks ml
             JOIN performance_schema.threads t ON t.thread_id = ml.owner_thread_id
             WHERE ml.object_name = %s AND ml.lock_status = 'GRANTED'
               AND t.processlist_id <> CONNECTION_ID()"""


def alter_when_free(cur, table, ddl, label, deadline_s=3600, poll_s=20):
    """Returns True once ddl has run, False if the deadline passes first."""
    cur.execute("SET SESSION lock_wait_timeout = 5")
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        cur.execute(HOLDERS, (table,))
        holders = cur.fetchall()
        if holders:
            pid, cmd, age, info = holders[0]
            print(f"    {label}: MDL 被 pid={pid} ({cmd}, {age}s) 占用 "
                  f"[{info}] — 等 {poll_s}s  已等 {(time.time()-t0)/60:.1f}m", flush=True)
            time.sleep(poll_s)
            continue
        try:
            cur.execute(ddl)
            print(f"    {label}: 已执行 (等待 {(time.time()-t0)/60:.1f}m)", flush=True)
            return True
        except pymysql.err.OperationalError as e:
            if e.args[0] != 1205:
                raise
            print(f"    {label}: 抢锁失败，10s 后重试", flush=True)
            time.sleep(10)
    print(f"    {label}: 超时放弃，稍后重跑即可", flush=True)
    return False


def column_exists(cur, table, column):
    cur.execute("""SELECT COUNT(*) FROM information_schema.columns
                   WHERE table_schema='field' AND table_name=%s AND column_name=%s""",
                (table, column))
    return bool(cur.fetchone()[0])
