import os
import sys

# Add the parent directory to the system path to find app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the server from your app.py and rename it to 'app' for Vercel
from app import server as app
