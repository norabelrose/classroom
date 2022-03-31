<script lang="ts">
    export let showRedundant: boolean = true;
</script>

<main>
    <h1>View Settings</h1>
    <label for="redundant" class="setting"
    title="Redundant preferences are those that are not included in the most compact representation of the preference ordering, known as the 'transitive reduction.' These preferences can be deduced via the transitive property from other preferences.">
        Show redundant preferences
    </label>
    <input type="checkbox" id="redundant" bind:checked={showRedundant}/>
</main>

<style>
    main {
        width: 12rem;
    }
    h1 {
        font-size: 1rem;
        font-weight: bold;
        padding-top: 0;
        text-align: center;
    }
    .setting {
        font-size: small;
    }
</style>