<script lang="ts">
    import { GraphController } from './graph_controller';
    import { rewardType, showEnvRewards, showIndifferences, showRedundant } from '../stores';

    // Mildly annoying that we have to do this to get the $ syntax to work
    const hasEnvRewards = GraphController.hasEnvRewards;
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
    <div>
        <label for="envRewards" class="setting"
        title="Whether to display the 'true' rewards provided by the environment next to each node.">
            Show environment rewards
        </label>
        <input type="checkbox" id="envRewards" bind:checked={$showEnvRewards} disabled={!$hasEnvRewards}/>
    </div>
    <div>
        <label for="reward-type" title="Method to use for inferring rewards from preferences">Inferred rewards</label>
        <select name="reward-type" id="reward-type" bind:value={$rewardType}>
            <option title="The probability that a clip is preferred to another randomly selected clip" value="borda">Borda score</option>
            <option value="bradley-terry">Bradley-Terry</option>
            <option value="thurstone">Thurstone</option>
            <option value="none">None</option>
        </select>
    </div>
</div>

<style>
    .header {
        font-size: 1rem;
        padding-top: 0;
        text-align: center;
    }
    .setting {
        font-size: small;
    }
</style>