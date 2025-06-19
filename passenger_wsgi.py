import os
import sys

# Add the application's directory to the Python path
# This ensures that 'app.py' and other modules can be found
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app instance from your app.py file
# The server will run this 'application' object
from app import app as application 