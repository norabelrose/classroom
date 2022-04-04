<script lang="ts">
    export let items: string[] = [];
</script>

<footer class="bar">
    {#each items as item}
        <div class="bar-item">{item}</div>
    {/each}
</footer>

<style>
    .bar {
        background-color: var(--nav-color);
        border-top: 1px solid rgb(55, 59, 65);
        box-shadow: 0rem -0.2rem 0.3rem var(--shadow-color);
        display: flex;
        font-size: small;
        height: 1.625rem; /* 22px */
        vertical-align: middle;
        width: 100%;
    }
    .bar-item {
        margin: 0.25rem 1rem 0.25rem 1rem;
    }
</style>