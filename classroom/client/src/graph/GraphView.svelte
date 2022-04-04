<script lang="ts">
    import { attachFloaterOnSelect } from './attach_floater';
    import { autoUpdateColorScheme, graphStyles } from './graph_styles';
    import cytoscape from 'cytoscape';
    import dagre from 'cytoscape-dagre';
    import { globalSocket } from '../rpc_socket';
    import { onDestroy, onMount } from 'svelte';
    import StatusBar from '../StatusBar.svelte';
    cytoscape.use(dagre);

    let cleanupFn: () => void;
    let container: HTMLDivElement;
    let graph: cytoscape.Core;
    let statusItems: string[] = [];

    onMount(async () => {
        const {
            nodes, strictPrefs, indifferences,
            connectedNodes, totalNodes, numIndifferences, numPrefs, longestPath
        } = await globalSocket.call('getGraph');

        statusItems = [
            `Showing ${connectedNodes} of ${totalNodes} nodes`,
            `${numPrefs} strict prefs`,
            `${numIndifferences} indifferences`,
            `Longest path: ${longestPath} hops`,
        ];
        graph = cytoscape({
            container,
            elements: {
                nodes,
                edges: [...strictPrefs, ...indifferences]
            },
            
            layout: {
                name: 'dagre',
                fit: false,
                nodeDimensionsIncludeLabels: true,
            },

            // If you zoom out too far you may not be able to find the nodes
            minZoom: 0.2,
            maxZoom: 1,
            style: graphStyles,
        });

        attachFloaterOnSelect(graph, (node) => {
            const floater = document.createElement('iframe');
            floater.src = `/viewer_html/${node.id()}`;
            Object.assign(
                floater.style,
                { border: 'none', 'border-radius': '10px' }
            )
            return floater;
        });
        cleanupFn = autoUpdateColorScheme(graph);
    });
    
    onDestroy(() => { if (cleanupFn) cleanupFn() });
</script>

<div bind:this={container} style="flex-grow: 1;">
    <!-- Cytoscape UI goes here -->
</div>
<StatusBar items={statusItems}/>