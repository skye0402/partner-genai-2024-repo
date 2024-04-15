import os
import logging
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta, UTC

from langchain_community.llms.vllm import VLLMOpenAI

from hana_ml import ConnectionContext
from hdbcli.dbapi import Connection

class AICoreHandling:
    def __init__(self) -> None:
        """ Initialize the class """
        self.resource_group = os.environ.get("AICORE_RESOURCE_GROUP")
        self.token=None
        self.token_expires_at=None
        self.token_url=os.environ.get("AICORE_AUTH_URL")
        self.client_id=os.environ.get("AICORE_CLIENT_ID")
        self.client_secret=os.environ.get("AICORE_CLIENT_SECRET")
        self.deployment=os.environ.get("DEPLOYMENT_NAME")
        self.dep_api_path=os.environ.get("DEPLOY_PATH")
        self.aicore_base=os.environ.get("AICORE_API_BASE")
              
    def fetch_token(self):   
        """ Fetch OAuth2 token for AI Core Authentication """
        # Check if the buffered token is still valid
        if self.token and self.token_expires_at > datetime.now(UTC):
            return f"Bearer {self.token['access_token']}"    
        # Prepare the data for the token request
        data = {
            'grant_type': 'client_credentials'
        }    
        try:
            # Make the POST request to the token endpoint with the client credentials
            response = requests.post(
                self.token_url,
                data=data,
                auth=HTTPBasicAuth(self.client_id, self.client_secret)
            )        
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the token from the response
                self.token = response.json()
                
                # Buffer the token and calculate its expiration time
                expires_in=self.token.get('expires_in', 3600)  # Default to 1 hour if not present
                self.token_expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)            
                return f"Bearer {self.token['access_token']}" 
            else:
                # Handle error response
                print(f"Failed to fetch token: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            # Handle any exceptions that occur during the request
            print(f"An error occurred while fetching the token: {e}")
            return None
        
    def get_baseurl(self)->str:
        """ Retrieves the AI Core deployment URL """
        # Request an access token using client credentials
        access_token = self.fetch_token()        
        headers = {
            'Authorization': access_token,
            'AI-Resource-Group': self.resource_group
        }
        res = requests.get(self.aicore_base+self.dep_api_path, headers=headers)
        j_data = res.json()
        for resource in j_data["resources"]:
            if resource["scenarioId"] == self.deployment:
                if resource["deploymentUrl"] == "":
                    print(f"Scenario '{self.deployment}' was found but deployment URL was empty. Current status is '{resource['status']}', target status is '{resource['targetStatus']}'.")
                else:
                    print(f"Scenario '{self.deployment}': Plan '{resource['details']['resources']['backend_details']['predictor']['resource_plan']}', modfied at {resource['modifiedAt']}.")
                return f"{resource['deploymentUrl']}/v1"
        
def get_llm_model(ai_core: AICoreHandling, temperature=0.0, top_p=0.95, max_tokens=3500, do_streaming=False)->any:
    """ Serve the required LLM chat model """ 
    return VLLMOpenAI(
        default_headers={"Authorization": ai_core.fetch_token(), "AI-Resource-Group": ai_core.resource_group},
        openai_api_key="EMPTY",
        openai_api_base=ai_core.get_baseurl(),
        model_name="TheBloke/Starling-LM-7B-alpha-AWQ",
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        streaming=do_streaming
    )

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
        logging.debug(conn_context.hana_version())
        logging.debug(conn_context.get_current_schema())
    except Exception as e:
        logging.error(f'Error when opening connection to HANA Cloud DB with host: {conn_params["host"]}, user {conn_params["user"]}. Error was:\n{e}')
    finally:    
        return connection