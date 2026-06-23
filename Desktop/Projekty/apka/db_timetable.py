import psycopg
import pandas as pd
import streamlit as st

def get_conn():
    return psycopg.connect(st.secrets["SUPABASE_DB_URL"])

# 🔥 usuń stare i dodaj nowe
def replace_timetable(df: pd.DataFrame, run_name="upload"):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # delete wszystko
                cur.execute("DELETE FROM timetable_entries;")
                cur.execute("DELETE FROM timetables;")

                # insert nowy timetable
                cur.execute(
                    "INSERT INTO timetables (run_name, active) VALUES (%s, true) RETURNING id",
                    (run_name,)
                )
                timetable_id = cur.fetchone()[0]

                rows = []
                for _, r in df.iterrows():
                    rows.append((
                        timetable_id,
                        str(r.get("Day", "")),
                        str(r.get("Hour", "")),
                        str(r.get("Students", "")),
                        str(r.get("Subject", "")),
                        str(r.get("Teacher", "")),
                        str(r.get("Room", "")),
                        None
                    ))

                cur.executemany("""
                    INSERT INTO timetable_entries
                    (timetable_id, day, hour, student_group, subject, teacher, room, raw)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, rows)

    finally:
        conn.close()


# 🔥 load do DataFrame (do Twojego ML)
def load_timetable_df():
    conn = get_conn()
    try:
        df = pd.read_sql("""
            SELECT day, hour, student_group, subject, teacher, room
            FROM timetable_entries
        """, conn)
        return df
    finally:
        conn.close()