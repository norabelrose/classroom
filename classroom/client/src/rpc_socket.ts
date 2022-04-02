/* Simple class for asynchronous remote procedure calls over a WebSocket */
export class RpcSocket {
    private socket: WebSocket;
    private callbacks: { [id: number]: (data: any) => void } = {};
    private nextId: number = 0;
    private msgQueue: string[] = [];
    
    constructor(url: string) {
        this.socket = new WebSocket(url);
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.id in this.callbacks) {
                this.callbacks[data.id](data.result);
                delete this.callbacks[data.id];
            }
        };
        this.socket.onopen = () => {
            this.msgQueue.forEach(msg => this.socket.send(msg));
            this.msgQueue = [];
        };
    }
    
    async call(method: string, params: any = null): Promise<any> {
        const id = this.nextId++;
        const msg = JSON.stringify({ id, method, params });
        const promise = new Promise((resolve, _) => this.callbacks[id] = resolve);
        
        // If the socket is connected, send right away
        if (this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(msg);
        }
        // Otherwise, queue the message
        else {
            this.msgQueue.push(msg);
        }
        return promise;
    }
}

export const globalSocket = new RpcSocket(`ws://${location.host}/feedback`);