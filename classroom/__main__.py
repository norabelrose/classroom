from argparse import ArgumentParser
from pathlib import Path
from .gui_server import GuiServer


def main():
    parser = ArgumentParser(description="Run a CherryPy server to serve the Classroom GUI.")
    parser.add_argument('--allowed-ips', nargs='*', type=str, help="List of allowed remote IPs.")
    parser.add_argument('--logdir', type=Path, help="Path to the experiment directory.")
    parser.add_argument('--num-tokens', type=int, default=0, help="Number of tokens to generate for authentication.")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the CherryPy server on.")
    args = parser.parse_args()

    app = GuiServer(
        allowed_ips=args.allowed_ips,
        experiment_dir=args.logdir,
        num_tokens=args.num_tokens,
        port=args.port
    )
    if args.num_tokens:
        print("You can use one of the following URLs:")
        for token in app.tokens:
            print(f"\thttp://localhost:{args.port}/token/{token}")
    
    app.serve_forever()# run()


if __name__ == '__main__':
    main()
