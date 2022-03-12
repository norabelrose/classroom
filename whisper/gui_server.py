from flask import Flask, render_template
from pathlib import Path
from typing import Optional
import os
import secrets


INSTRUCTIONS = [
    "You will see a task description and two videos of agents attempting to perform that task",
    "Decide which agent is coming closer to performing that task well",
    "Press the appropriate button",
]


class GuiServer:
    """Wrapper class for a Flask app for gathering human feedback."""
    def __init__(
            self,
            port: int = 5000,
            instructions: list[str] = INSTRUCTIONS,
            debug: bool = False,
            experiment_dir: Optional[Path] = None,
            num_tokens: int = 0,    # Number of user tokens to generate; 0 means no token authentication
        ):
        self.experiment_dir = experiment_dir or Path.cwd()
        self.port = port

        root = Path(__file__).parent
        app = Flask("Whisper", static_folder=root / 'static', template_folder=str(root / 'templates'))
        app.debug = debug

        # Simply generate a new secret key each time the app is run. This makes things simple and secure,
        # and it's sufficient for our purposes because we never actually need to store signed cookies that
        # last longer than the lifetime of the app.
        app.secret_key = secrets.token_hex()

        # Generate URL tokens for users to use to authenticate themselves
        self.tokens = {secrets.token_urlsafe(16) for _ in range(num_tokens)}

        @app.route('/token/<token>' if num_tokens > 0 else '/')
        def home(token: Optional[str] = None):
            # If we have tokens, check that the token is valid
            if num_tokens > 0 and token not in self.tokens:
                return "Invalid token"
            
            # Get list of experiments & runs
            runs = {}
            for path in self.experiment_dir.iterdir():
                if not path.is_dir():
                    continue

                name = path.name
                if name.startswith('.'):
                    continue

                runs[name] = [
                    child.name
                    for child in path.iterdir()
                    if child.is_dir()
                ]
            
            return render_template(
                'home.html', experiments=runs, instructions=instructions
            )
        
        self.app = app
    
    def run(self):
        """Run the Flask app."""
        self.app.run(port=self.port)
