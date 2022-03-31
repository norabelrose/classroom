<script lang="ts">
    import { selectedRun } from './stores';

    type Folder = {
        name: string;
        children: Folder[];
    };
    export let expanded: boolean;
    export let folder: Folder;
</script>

<li class="control experiment">
    <!-- Only show the caret for nonempty folders/runs -->
    {#if folder.children.length > 0}
        <span class="caret" class:expanded on:click={() => expanded = !expanded}>
            <svg><use href="/buttons.svg#caret"/></svg>
        </span>
    {/if}
    <span class:selected={$selectedRun === folder.name} on:click={() => {
        // Only allow the folder to be selected if it's a leaf node
        if (folder.children.length === 0) {
            $selectedRun = folder.name;
        }
    }}>
        {folder.name}
    </span>
    {#if expanded}
        <div class="indent">
            {#each folder.children as elem}
                <svelte:self bind:folder={elem} expanded={false}/>
            {/each}
        </div>
    {/if}
</li>

<style>
    .caret {
        display: inline-block;
        height: 1rem;
        margin-right: 0.4rem;
        transform: rotate(-90deg);
        vertical-align: bottom;
        width: 1rem;
    }
    /* Rotate the caret when the checkbox is checked & animate the transition */
    .expanded {
        transform: none;
        transition: ease-in 0.2s;
    }
    .indent {
        margin-left: 1.75rem;
    }
    .selected {
        color: var(--highlight-color);
        font-weight: bold;
    }
</style>