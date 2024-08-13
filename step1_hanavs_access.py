from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from hana_ml import ConnectionContext
from hdbcli.dbapi import Connection

import os, logging

def get_hana_connection(conn_params: dict)->Connection:
    """ Connect to HANA Cloud """
    connection = None  
    try:
        conn_context = ConnectionContext(
            address = conn_params["host"],
            port = 443,
            user = conn_params["user"],
            password= conn_params["password"],
            encrypt= True
        )    
        connection = conn_context.connection
        logging.info(f"Connected to HANA Cloud version {conn_context.hana_version()}.")
        logging.info(f"Schema is '{conn_context.get_current_schema()}'.")
    except Exception as e:
        logging.error(f'Error when opening connection to HANA Cloud DB with host: {conn_params["host"]}, user {conn_params["user"]}. Error was:\n{e}')
    finally:    
        return connection
    
def main()->None:
    """ Let's check the connection - that's all what it does """    
    log_level = int(os.environ.get("APPLOGLEVEL", logging.INFO))
    if log_level < 10: log_level = 40
    logging.basicConfig(level=log_level,)
    
    # Get the HANA Cloud data from the environment
    hana_cloud = {
        "host": os.getenv("HANA_DB_ADDRESS"),
        "user": os.getenv("HANA_DB_USER"),
        "password": os.getenv("HANA_DB_PASSWORD") 
    }
    if (hana_cloud["host"]==None or hana_cloud["user"]==None or hana_cloud["password"]==None):
        logging.error("One or more of the host/user/password environment variables is not set. Did you maintain the '.env' file?")
        exit()
    logging.info(f"Connecting to SAP HANA Cloud host {hana_cloud.get('host')} with user {hana_cloud.get('user')}...")
    hana_conn = get_hana_connection(conn_params=hana_cloud)
    try:
        hana_conn.close()
        logging.info("Connection test was successful.") 
    except Exception as e:
        logging.error(f"Error occurred: {e}")   
    
if __name__ == "__main__":
    main()