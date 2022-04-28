import cxtmenu from 'cytoscape-cxtmenu';
import cytoscape from "cytoscape";
import dagre from 'cytoscape-dagre';
import edgehandles from 'cytoscape-edgehandles';
import type { Core, NodeSingular, SingularData } from 'cytoscape';
import { graphStyles } from './graph_styles';
import { RpcSocket } from "../rpc_socket";
import { get, writable, Writable } from 'svelte/store';


export type GraphEdit = {
    action: 'add' | 'remove' | 'invert',
    source: string,
    target: string,
    attrs?: Object
};
export type Pref = '>' | '<' | '=';
export type QueryPair = {left: string, right: string};

export let undoStack: Writable<GraphEdit[]> = writable([]);
export let redoStack: Writable<GraphEdit[]> = writable([]);

export let statusBarItems: Writable<string[]> = writable([]);

/** `GraphController` encapsulates all glue code needed to ensure that the Cytoscape
 * visualization of the preference graph and the `PrefGraph` data structure kept on
 * the server are in sync. It manages an undo-redo stack, and provides methods to
 * commit changes to the graph. */
export namespace GraphController {
    let graph: cytoscape.Core;
    let unmountCleanup = () => {};

    const socket = new RpcSocket(`ws://${location.host}/feedback`);

    /** Load the graph from the server. */
    export async function init() {
        if (graph) return;
        
        const { nodes, strictPrefs, indifferences } = await socket.call('getGraph');

        // Activate Cytoscape extensions
        cytoscape.use(cxtmenu);
        cytoscape.use(dagre);
        cytoscape.use(edgehandles);

        // Create the graph
        graph = cytoscape({
            elements: {
                nodes,
                edges: [...strictPrefs, ...indifferences]
            },
    
            // If you zoom out too far you may not be able to find the nodes
            minZoom: 0.2,
            maxZoom: 1,
            style: graphStyles,
        });

        // Configure extensions
        graph.edgehandles({
            // Disallow parallel edges and self-loops
            canConnect: (source, target) => !source.edgesWith(target) && !source.same(target),
        });
        // @ts-expect-error
        graph.on('ehcomplete', (event, source, target, addedEdge) => {
            socket.call('add_pref', { left: source.id(), right: target.id(), pref: '>' });
        });

        // Register the graph to have its color scheme automatically updated to match the browser preference
        const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');
        const updateTheme = (e: MediaQueryListEvent) => {
            const contrastColor = e.matches ? 'white' : 'black';
            
            graph.style()   // @ts-expect-error
                .selector('*').style({ color: contrastColor })
                .selector('edge').style({ 'line-color': contrastColor })
                .selector('edge[strict]').style({ 'target-arrow-color': contrastColor })
                .update();
        };
        themeQuery.addEventListener('change', updateTheme);
    }

    export function updateStatusBar() {
        socket.call('getStats').then(stats => {
            const { connectedNodes, totalNodes, longestPath, numPrefs, numIndifferences } = stats;
            statusBarItems.set([
                `${numPrefs} strict prefs, ${numIndifferences} indifferences`,
                `${connectedNodes} evaluated of ${totalNodes} clips`,
                `Longest path: ${longestPath} hops`,
            ]);
        });
    }

    function _apply(edit: GraphEdit) {
        const { action, source, target, attrs } = edit;
        switch (action) {
        case 'add':
            // Update the server
            socket.call('add_pref', { source, target, attrs });

            // Update Cytoscape view *if necessary*- this is a no-op
            // if we haven't yet received the graph from the server
            if (!graph) return;

            _addEdge(source, target, attrs);
            break;
        case 'remove':
            // Update Cytoscape view
            graph.remove(`#${source}-${target}`);

            // Update the server
            socket.call('remove_pref', { source, target });
            break;
        case 'invert':
            break;
        }
    }

    function _addEdge(source: string, target: string, attrs?: Object) {
        graph.startBatch();

        // Add the source if it's not already there
        const srcNode = graph.$id(source);
        if (srcNode.removed())
            srcNode.restore();
        else if (!srcNode.inside())
            graph.add({ group: 'nodes', classes: 'clip', data: { id: source } });
        
        // Add the target if it's not already there
        const tgtNode = graph.$id(target);
        if (tgtNode.removed())
            tgtNode.restore();
        else if (!tgtNode.inside())
            graph.add({ group: 'nodes', classes: 'clip', data: { id: target } });

        // Add the actual edge
        const id = `${source}-${target}`;
        const edge = graph.$id(id);
        if (edge.removed())
            edge.restore();
        else if (!edge.inside())
            graph.add({
                group: 'edges',
                classes: 'pref',
                data: { source, target, id, ...attrs }
            });
        
        graph.endBatch();
    }

    export function commit(edit: GraphEdit) {
        undoStack.update(old => [edit, ...old]);
        _apply(edit);
        redoStack.update(_ => []);
    }
    export async function currentPair(): Promise<QueryPair> {
        return await socket.call('clips');
    }
    export async function commitFeedback(pref: Pref, query: QueryPair): Promise<QueryPair> {
        const { left, right } = query;
        const [source, target] = pref === '<' ? [right, left] : [left, right];

        undoStack.update(old => [
            { action: "add", source, target, attrs: { strict: pref !== '=' } },
            ...old
        ]);
        redoStack.update(_ => []);

        if (graph)
            _addEdge(source, target, { strict: pref !== '=' });

        // Update the server
        const res = await socket.call('add_pref', { source, target, weight: pref !== '=' ? 1 : 0 });
        updateStatusBar();
        return res;
    }

    export function mount(container: HTMLDivElement) {
        graph.mount(container);

        graph.on('select', 'node', (event) => {
            const floater = document.createElement('iframe');
            floater.src = `/viewer_html/${event.target.id()}`;
            Object.assign(
                floater.style,
                { border: 'none', 'border-radius': '10px' }
            )

            setTimeout(() => {
                unmountCleanup = attachFloater(graph, floater, event.target);
            }, 200);
        });
        graph.on('unselect', 'node', _ => {
            unmountCleanup();
            unmountCleanup = () => {};
        });

        graph.cxtmenu({
            selector: 'edge',
            commands: (ele) => {
                return [
                    {
                        content: 'Remove',
                        select: (ele: SingularData) => {
                            graph.remove(`#${ele.id()}`);
                            socket.call('remove_pref', ele.data());
                        }
                    },
                    // Invert the direction of the preference. This is only enabled for strict
                    // preferences, not indifferences.
                    {
                        content: 'Invert',
                        enabled: ele.data('strict') === true,
                        select: (ele: SingularData) => socket.call('invertPref', ele.data())
                    },
                ];
            }
        });
        graph.layout({
            name: 'dagre',
            fit: false,
            nodeDimensionsIncludeLabels: true,
        }).run();
    }
    export function unmount() {
        graph.unmount();
        unmountCleanup();
    }

    export function undo() {
        const last = get(undoStack)[0];
        undoStack.update(old => old.slice(1));

        _apply({ ...last, action: last.action === 'add' ? 'remove' : 'add' });
        redoStack.update(old => [last, ...old]);
    }

    export function redo() {
        const last = get(redoStack)[0];
        redoStack.update(old => old.slice(1));

        _apply(last);
        undoStack.update(old => [last, ...old]);
    }

    // 'Attach' a floating HTML element to a Cytoscape node, ensuring that its position
    // keeps in sync with the node's position.
    function attachFloater(graph: Core, attached: HTMLElement, node: NodeSingular, offset = {x: 0, y: 0}): () => void {
        function update() {
            const { x: centerX, y: centerY } = node.renderedPosition();
            const width = node.renderedOuterWidth();
            const height = node.renderedOuterHeight();
            const x = centerX - width / 2;
            const y = centerY - height / 2;

            Object.assign(attached.style, {
                height: `${height}px`,
                width: `${width}px`,
                left: `${x + offset.x}px`,
                top: `${y + offset.y}px`,
            });
        }

        Object.assign(attached.style, {
            position: 'absolute',
            'z-index': 1000,
        });
        update();
        graph.container()!.appendChild(attached);

        // Now add a listener to update the position whenever the node is moved
        graph.on('position', 'node', update);
        graph.on('viewport', update);

        return () => {
            graph.removeListener('position', 'node', update);
            graph.removeListener('viewport', update);
        };
    }
}