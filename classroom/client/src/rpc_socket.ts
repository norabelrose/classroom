import ReconnectingWebSocket from 'reconnecting-websocket';

/* Simple class for asynchronous remote procedure calls over a WebSocket.
 * We use this instead of `fetch()` for API calls to the server mainly because
 * it allows us to write the server-side code in a really clean way using
 * an async-await coroutine. */
export class RpcSocket {
    private socket: ReconnectingWebSocket;
    private callbacks: { [id: number]: (data: any) => void } = {};
    private nextId: number = 0;
    
    constructor(url: string) {
        this.socket = new ReconnectingWebSocket(url);
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.id in this.callbacks) {
                // Turn server-side errors into client-side exceptions
                if (data.error) {
                    throw new Error(data.error);
                } else {
                    this.callbacks[data.id](data.result);
                }
                delete this.callbacks[data.id];
            }
        };
    }
    
    async call(method: string, params: any = null): Promise<any> {
        const id = this.nextId++;
        const msg = JSON.stringify({ id, method, params });
        const promise = new Promise((resolve, _) => this.callbacks[id] = resolve);
        
        // ReconnectingWebSocket handles buffering messages until the socket is open
        this.socket.send(msg);
        return promise;
    }
}

export const globalSocket = new RpcSocket(`ws://${location.host}/feedback`);