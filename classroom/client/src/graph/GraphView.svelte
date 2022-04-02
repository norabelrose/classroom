<script lang="ts">
    import { attachFloaterOnSelect } from './attach_floater';
    import { autoUpdateColorScheme, graphStyles } from './graph_styles';
    import cytoscape from 'cytoscape';
    //import { Jumper } from 'svelte-loading-spinners';
    import { globalSocket } from '../rpc_socket';
    import { onDestroy, onMount } from 'svelte';

    let cleanupFn: () => void;
    let container: HTMLDivElement;
    let graph: cytoscape.Core;

    onMount(async () => {
        graph = cytoscape({
            container,
            elements: await globalSocket.call('getGraph'),

            layout: { name: 'grid' },

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