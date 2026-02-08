import requests
import json

BASE = "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin"

def safe_get(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def safe_post(url, payload):
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()

if __name__ == '__main__':
    print("Fetching databases...")
    dbs = safe_get(BASE)
    print(f"Databases returned: {len(dbs)}")
    # pick first list-type db
    first_db = None
    for item in dbs:
        if item.get('type') == 'l':
            first_db = item
            break
    if not first_db:
        print('No database found')
        raise SystemExit(1)
    # Find a database that contains list-type tables
    db_id = None
    selected_db = None
    for item in dbs:
        if item.get('type') != 'l':
            continue
        candidate_id = item['id']
        try:
            tables = safe_get(f"{BASE}/{candidate_id}")
        except Exception as e:
            print('Skipping', candidate_id, 'due to', e)
            continue
        # tables can be a dict with error key if rate limited; ensure it's a list
        if not isinstance(tables, list):
            continue
        list_tables = [t for t in tables if t.get('type') == 'l']
        if list_tables:
            db_id = candidate_id
            selected_db = item
            break

    if not db_id:
        print('No database with tables found')
        raise SystemExit(1)

    print(f"Selected DB: {selected_db.get('text')} ({db_id})")
    first_table = list_tables[0]
    table_id = first_table['id']
    print(f"Selected table: {first_table.get('text')} ({table_id})")

    # fetch metadata
    meta = safe_get(f"{BASE}/{db_id}/{table_id}")
    vars = meta.get('variables', [])
    print(f"Variables in table: {len(vars)}")

    # build query choosing first value for each variable
    query = {"response": {"format": "json"}}
    for v in vars:
        code = v.get('code')
        opts = []
        if v.get('values'):
            opts = v.get('values')
        elif v.get('valueTexts') and isinstance(v.get('valueTexts'), dict):
            opts = list(v.get('valueTexts').keys())
        elif v.get('valueTexts') and isinstance(v.get('valueTexts'), list):
            opts = v.get('valueTexts')
        if opts:
            query[code] = [opts[0]]
    print('Query built with variables:', list(query.keys()))

    data = safe_post(f"{BASE}/{db_id}/{table_id}", query)
    if 'data' in data:
        print('Data retrieved, sample count:', len(data['data']))
    else:
        print('No data key in response')

    print('Done')
