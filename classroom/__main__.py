from argparse import ArgumentParser
from pathlib import Path
import warnings


if __name__ == '__main__':
    parser = ArgumentParser(description="Run the Classroom GUI server.")
    parser.add_argument('--allowed-ips', nargs='*', type=str, help="List of allowed remote IPs.")
    parser.add_argument('--database', type=Path, help="Path to the database directory.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument(
        '--usernames', nargs='*', type=str, default=(), help="Usernames for which to generate tokens."
    )
    parser.add_argument('--port', type=int, default=8000, help="Port to run the server on.")
    args = parser.parse_args()

    # Prevent annoying DeprecationWarnings from inside Sanic
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='the imp module')

    from .gui_app import app
    app.config.update(**vars(args))
    app.run(
        auto_reload=args.debug,
        debug=args.debug,
        host='localhost' if not args.allowed_ips else '0.0.0.0',
        port=args.port,
        fast=True   # Use all available cores when needed
    )
