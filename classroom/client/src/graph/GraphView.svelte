<script lang="ts">
    import { GraphController } from './graph_controller';
    import { onDestroy, onMount } from 'svelte';

    let container: HTMLDivElement;
    onMount(async () => {
        await GraphController.init();
        GraphController.mount(container);
    });

    onDestroy(GraphController.unmount);
</script>

<div bind:this={container} style="flex-grow: 1;">
    <!-- Cytoscape UI goes here -->
</div>