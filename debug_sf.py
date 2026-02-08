import requests, json
BASE = "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin"
for db in ['matk','khi','tyonv','vaerak','vaenn']:
    try:
        r = requests.get(f"{BASE}/{db}", timeout=10)
        print('DB', db, 'status', r.status_code)
        data = r.json()
        print('Type:', type(data))
        if isinstance(data, list):
            print('Len:', len(data))
            if len(data)>0:
                print('First item keys:', list(data[0].keys()))
        else:
            print('Keys:', list(data.keys())[:10])
        print('-'*40)
    except Exception as e:
        print('Error', db, e)
