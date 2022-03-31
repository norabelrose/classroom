<script lang="ts">
    import { availableRuns } from './stores';
    import FolderList from './FolderList.svelte';
</script>

<ul>
    {#each $availableRuns as folder}
        <li class="control">
            <FolderList folder={folder} expanded={false}/>
        </li>
    {/each}
</ul>

<style>
    ul {
        line-height: 1.8;
        list-style-type: none;
    }
</style>