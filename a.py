import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
#make sure agent is working
def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-flash-latest"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="hi tell about dogs")],
        ),
    ]
    generate_content_config = types.GenerateContentConfig()

    # --- RETRY LOGIC START ---
    max_retries = 5
    base_delay = 5  # Start waiting 5 seconds, then double it
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting generation (Try {attempt + 1})...")
            
            # The API call
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                print(chunk.text, end="")
            
            # If successful, break the loop
            break 
            
        except errors.ClientError as e:
            # Check if it is a 429 Resource Exhausted error
            if e.code == 429:
                print(f"\nRate limit hit. Waiting {base_delay} seconds...")
                time.sleep(base_delay)
                base_delay *= 2  # Exponential backoff (wait longer next time)
            else:
                # If it's a different error, crash as normal
                raise e
    # --- RETRY LOGIC END ---

if __name__ == "__main__":
    generate()
