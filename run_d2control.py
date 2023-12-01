from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio

# Create a configuration object for Hypercorn
config = Config()
config.bind = ["0.0.0.0:5000"]  # IP and port to bind
config.worker_class = "asyncio"  # Set worker class to asyncio
config.loglevel = 'debug'  # Enable debug logging

# Import your application
from d2control import app

# Asynchronously serve the application with the specified configuration
async def run():
    await serve(app, config)

# Run the async function with asyncio
if __name__ == "__main__":
    asyncio.run(run())
