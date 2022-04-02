<script lang="ts">
	import ComparisonView from './ComparisonView.svelte';
	import GraphView from './graph/GraphView.svelte';
	import Modal from './Modal.svelte';
	import NavBar from './NavBar.svelte';
	import StatusBar from './StatusBar.svelte';
	import { selectedTab } from './stores';

	let showingHelp = false;
</script>

<div id="content">
	<NavBar bind:showingHelp={showingHelp}/>

	{#if $selectedTab === 'compare'}
		<ComparisonView />
	{:else}
		<GraphView />
	{/if}

	<StatusBar/>
</div>

<Modal bind:visible={showingHelp}>
	<h2 slot="header">Instructions</h2>
	<ul slot="content">
		<li>You will see a task description and two videos of agents attempting to perform that task.</li>
		<li>Decide which agent is coming closer to performing that task well.</li>
		<li>Press the appropriate button.</li>
	</ul>
</Modal>


<style>
	#content {
		display: flex;
		flex-direction: column;
		height: 100%;
		width: 100%;
	}
</style>