from argparse import ArgumentParser
from .gui_server import GuiServer


def main():
    parser = ArgumentParser(description="Run a Flask server to serve the RL-Teacher GUI.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode.")
    parser.add_argument('--logdir', type=str, help="Path to the experiment directory.")
    parser.add_argument('--num-tokens', type=int, default=0, help="Number of tokens to generate for authentication.")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the Flask server on.")
    args = parser.parse_args()

    app = GuiServer(debug=args.debug, experiment_dir=args.logdir, num_tokens=args.num_tokens, port=args.port)
    if args.num_tokens:
        print("You can use one of the following URLs:")
        for token in app.tokens:
            print(f"\thttp://localhost:{args.port}/token/{token}")
    
    app.run()


if __name__ == '__main__':
    main()
