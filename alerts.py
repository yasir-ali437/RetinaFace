import requests
from datetime import datetime, timedelta
import logging
import os
import argparse


parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("--day", default=2, type=int, help="Which day's alerts to download and process")
args = parser.parse_args()
alerts_day = args.day

# Get alerts date using alerts day
alerts_date = datetime.now() - timedelta(days=alerts_day)
alerts_date = alerts_date.strftime('%Y-%m-%d')

LOG_FILE = "./logs/get_jeddah_alert.log"

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

def get_auth_token():
    url = "https://dsw-bk.core9ventures.com/api/Users/authenticate"
    log.info(f"Authenticate URL: {url}")

    # Payload
    payload = {
    "username": "AI_MODEL_S",
    "password": "MOdel@25"  
    }

    log.info(f"Payload: {payload}")

    response = requests.post(url, json=payload)
    #print("Response: ", response)

    if response.status_code == 200:
        data = response.json()
        #print(data)

        token = data.get('token')
        #print(token)

        return token
    
# Authorization token
AUTH_TOKEN = get_auth_token()

# API endpoint
url = "https://dsw-bk.core9ventures.com/api/Alert/GetAlertDataAnalysisV1"

# Headers
headers = {
    "Authorization": f"{AUTH_TOKEN}",
    "Content-Type": "application/json"
}


#date = '2025-04-21'
base_date = datetime.now() - timedelta(days=alerts_day)

start_time = base_date.replace(hour=15, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")
end_time = base_date.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%d %H:%M:%S")

# Payload
# "makeCalls", "smoke", "fatigueDriving", "notWearingSeatBeltAlarm", "distractedDriving", "driverAnomaly"
payload = {
    "alarmType": [
        "driverAnomaly"
    ],
    "startTime": start_time,
    "endTime": end_time,
    "orderBy": "ASC"
}

log.info(f"URL: {url}")
log.info(f"Payload: {payload}")

# Base URL to prepend to filePath
IMAGE_BASE_URL = "http://87.237.226.169:20003/"

# Directory to save downloaded files
SAVE_DIR = r"/home/adlytic/Yasir Adlytic/Dataset/New_Alerts/driverAnomaly"
os.makedirs(SAVE_DIR, exist_ok=True)

try:
    # Make the POST request
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        #print("Data: ", data)

        alerts = data.get("result")
        #print(alerts)
        log.info(f"Total alerts found: {len(alerts)}")

        downloaded_alerts = 0

        for alert in alerts:
            #print(alert)
            license_num = alert.get("licenseNum")
            alert_no = str(alert.get("vaId"))
            file_paths = alert.get("filePaths", [])
            alert_time = alert.get("startTime")

            if not alert_no or not file_paths or not alert_time:
                log.info(f"Alert has no vaId, filePath or startTime value: {alert_no}")
                continue

            if "J" in license_num:
                log.info(f"Alert: {alert_no}, License num: {license_num}")

                dt = datetime.strptime(alert_time, "%Y-%m-%d %H:%M:%S")
                date_str = dt.strftime("%Y-%m-%d")
                timestamp_str = dt.strftime("%Y%m%d_%H%M%S")

                # Folder path: seatbelt/YYYY-MM-DD/alarmNo/
                folder_path = os.path.join(SAVE_DIR, date_str, alert_no)
                os.makedirs(folder_path, exist_ok=True)

                for idx, file_path in enumerate(file_paths, start=1):
                    # Get extension
                    ext = os.path.splitext(file_path)[-1].lower()

                    # Skip images
                    if ext == ".jpg":
                        continue
                    
                    suffix = "_video" if ext == ".mp4" else f"_{idx}"

                    new_filename = f"{alert_no}_{timestamp_str}{suffix}{ext}"
                    full_url = IMAGE_BASE_URL + file_path
                    save_path = os.path.join(folder_path, new_filename)

                    if os.path.exists(save_path):
                        continue           

                    # Download the file
                    try:
                        img_response = requests.get(full_url)
                        if img_response.status_code == 200:
                            with open(save_path, 'wb') as f:
                                f.write(img_response.content)

                        else:
                            log.error(f"Failed to download file. Status code: {img_response.status_code}")
                    except Exception as e:
                        log.error(f"Error downloading file: {e}")

                downloaded_alerts += 1
            
        log.info(f"Total alerts downloaded: {downloaded_alerts}")

    else:
        log.error(f"API request failed. Status code: {response.status_code}")
        log.error(f"Response: {response.text}")

except Exception as e:
    log.error(f"Exception during getting alerts: {e}")