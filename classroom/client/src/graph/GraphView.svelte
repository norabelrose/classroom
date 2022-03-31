<script lang="ts">
    import cytoscape from 'cytoscape';
    import { attachFloaterOnSelect } from './attach_floater';
    import { autoUpdateColorScheme, graphStyles } from './graph_styles';
    import { onDestroy, onMount } from 'svelte';

    let cleanupFn: () => void;
    let graph: cytoscape.Core;
    // export let selectedNode: number | null = null;

    onMount(async () => {
        graph = cytoscape({
            container: document.getElementById('graph-container'),
            elements: await fetch('/graph').then(res => res.json()).catch(console.error) ?? [],
            layout: { name: 'grid' },

            // If you zoom out too far you may not be able to find the nodes
            minZoom: 0.2,
            maxZoom: 1,
            style: graphStyles
        });
        attachFloaterOnSelect(graph, (node) => {
            const floater = document.createElement('iframe');
            floater.src = `/viewer_html?node=${node.id()}`;
            Object.assign(
                floater.style,
                { border: 'none', 'border-radius': '10px' }
            )
            return floater;
        });
        cleanupFn = autoUpdateColorScheme(graph);
    });
    onDestroy(() => { if (cleanupFn) cleanupFn(); });
</script>

<div id="graph-container">
    {#if graph == null}
        <h1 style="text-align: center;">Loading graph...</h1>
    {/if}
    <!-- Cytoscape UI goes here -->
</div>

<style>
    #graph-container {
        width: 100%;
        height: calc(100% - 5.5rem);
    }
</style>