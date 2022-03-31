<script lang="ts">
    import { availableRuns, selectedRun } from './stores';
</script>

<footer class="bar">
    <div class="bar-item">
        {$availableRuns.length} available runs
    </div>
</footer>

<style>
    .bar {
        background-color: var(--nav-color);
        border-top: 1px solid rgb(55, 59, 65);
        box-shadow: 0rem -0.2rem 0.3rem var(--shadow-color);
        font-size: small;
        height: 1.625rem; /* 22px */
        vertical-align: middle;
        width: 100%;
    }
    .bar-item {
        margin: 0.25rem 1rem 0.25rem 1rem;
    }
</style>