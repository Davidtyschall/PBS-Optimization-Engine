"""
Supabase Client Configuration
PBS Optimization Engine
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Get credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Validate credentials exist
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "Missing Supabase credentials. "
        "Ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in .env file."
    )

# Create client instance
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def get_client() -> Client:
    """Return the Supabase client instance."""
    return supabase


def test_connection() -> bool:
    """Test the database connection by querying the users table."""
    try:
        # Simple query to test connection
        result = supabase.table("users").select("id").limit(1).execute()
        print(f"✓ Supabase connection successful")
        return True
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")
        return False


# Run test if executed directly
if __name__ == "__main__":
    test_connection()