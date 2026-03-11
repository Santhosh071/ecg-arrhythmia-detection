import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import wfdb
from config.config import MIT_BIH_PATH
from config.logger import get_logger
logger = get_logger(__name__)

RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

def download_mitbih():
    try:
        MIT_BIH_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to: {MIT_BIH_PATH}")
        logger.info(f"Total records: {len(RECORDS)}")
        for i, record in enumerate(RECORDS, 1):
            if (MIT_BIH_PATH / f"{record}.hea").exists():
                logger.info(f"[{i:02d}/48] Record {record} already exists — skipping")
                continue
            logger.info(f"[{i:02d}/48] Downloading record {record}...")
            wfdb.dl_database(
                db_dir="mitdb",
                dl_dir=str(MIT_BIH_PATH),
                records=[record]
            )
        logger.info("Download complete!")
        verify_download()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

def verify_download():
    try:
        hea_files = list(MIT_BIH_PATH.glob("*.hea"))
        logger.info(f"Records downloaded: {len(hea_files)} / 48")
        if len(hea_files) == 48:
            logger.info("All 48 records verified!")
        else:
            logger.warning(f"{48 - len(hea_files)} records missing — re-run to retry")
    except Exception as e:
        logger.error(f"Verification failed: {e}")

if __name__ == "__main__":
    download_mitbih()