import logging, warnings, os


# logging directory
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="{%(asctime)s) - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_filepath)])
logger = logging.getLogger("Sleep Quality Analysis ")

