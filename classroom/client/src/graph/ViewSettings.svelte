<script lang="ts">
    import { persistent } from '../stores';

    export let showIndifferences = persistent('showIndifferences', true);
    export let showRedundant = persistent('showRedundant', true);
</script>

<div class="container">
    <p class="header">View Options</p>
    <div>
        <label for="indifferences" class="setting">Show indifferences</label>
        <input type="checkbox" id="indifferences" bind:checked={$showIndifferences}/>
    </div>
    <div>
        <label for="redundant" class="setting"
        title="Redundant preferences are those that are not included in the most compact representation of the preference ordering, known as the 'transitive reduction.' These preferences can be deduced via the transitive property from other preferences.">
            Show redundant preferences
        </label>
        <input type="checkbox" id="redundant" bind:checked={$showRedundant}/>
    </div>
</div>

<style>
    div {
        width: 12rem;
    }
    .header {
        font-size: 1rem;
        padding-top: 0;
        text-align: center;
    }
    .setting {
        font-size: small;
    }
</style>