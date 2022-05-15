import ReconnectingWebSocket from 'reconnecting-websocket';
import { readable, Readable } from 'svelte/store';


const statusToString = new Map([
    [0, 'Connecting'],
    [1, 'Connected'],
    [2, 'Closing'],
    [3, 'Disconnected!'],
]);

/** Simple class for asynchronous remote procedure calls over a WebSocket.
 * We use this instead of `fetch()` for API calls to the server mainly because
 * it allows us to write the server-side code in a really clean way using
 * an async-await coroutine. */
export class RpcSocket {
    private socket: ReconnectingWebSocket;
    private callbacks: { [id: number]: (data: any) => void } = {};
    private nextId: number = 0;
    readonly readyState: Readable<string>;
    
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

        // Make the readyState an observable Svelte store
        const startState = statusToString.get(this.socket.readyState);
        this.readyState = readable(startState, set => {
            const update = () => set(statusToString.get(this.socket.readyState)!);

            this.socket.onopen = update;
            this.socket.onclose = update;
            this.socket.onerror = update;
            return () => {
                this.socket.onopen = null;
                this.socket.onclose = null;
                this.socket.onerror = null;
            }
        });
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
