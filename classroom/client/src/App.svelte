<script lang="ts">
	import ComparisonView from './ComparisonView.svelte';
	import GraphView from './graph/GraphView.svelte';
	import HamburgerButton from './HamburgerButton.svelte';
	import Modal from './Modal.svelte';
	import ViewSettings from './graph/ViewSettings.svelte';
	import { onMount } from 'svelte';
	import { selectedTab } from './stores';
	import tippy from 'tippy.js';
    import 'tippy.js/animations/scale.css';
    import 'tippy.js/dist/tippy.css';
    import 'tippy.js/themes/translucent.css';

	let popover: HTMLDivElement;
    let settingsButton: HTMLSpanElement;
	let showingHelp = false;
	let showingSidebar = false;

	onMount(() => {
        tippy(settingsButton, {
            // interactive: true,
            // allowHTML: true,
            animation: 'scale',
            arrow: true,
            content: popover,
            inertia: true,
            interactive: true,
            placement: 'bottom',
            trigger: 'click',
        });
    });
    
    // For the fullscreen button in the nav bar
	function toggleFullscreen() {
		if (document.fullscreenElement) {
			document.exitFullscreen();
		} else {
			document.documentElement.requestFullscreen();
		}
	}

    const KEYBINDINGS: Record<string, Function> = {
        c: () => $selectedTab = 'compare',
        v: () => $selectedTab = 'visualize',
        
        f: toggleFullscreen,
        h: () => showingHelp = !showingHelp,
    };

    function handleKeyDown(event: KeyboardEvent) {
        const fn = KEYBINDINGS[event.key];
        if (fn && !event.altKey && !event.ctrlKey && !event.shiftKey && !event.repeat) {
            event.preventDefault();
            fn();
        }
    }
</script>

<svelte:window on:keydown={handleKeyDown}/>

<div id="content">
	<header class="nav">
		<HamburgerButton bind:open={showingSidebar}/>
		<!-- Undo Button -->
		<span class="control nav-item" id="undo" title="Undo">
			<svg><use href="/buttons.svg#undo"/></svg>
		</span>
		<span class="spacer"></span>
		<div class="tab-container">
			<span class="control tab" class:tab-selected={$selectedTab === 'compare'}
				on:click={() => $selectedTab = 'compare'}
				title="Compare clips of AI behavior side-by-side">
				Compare
			</span>
			<span class="separator"/>
			<span class="control tab" class:tab-selected={$selectedTab === 'visualize'}
				on:click={() => $selectedTab = 'visualize'}
				title="Visualize all the preferences you've expressed as a graph">
				Visualize
			</span>
		</div>
		<span class="spacer"></span>
		<!-- Settings Icon -->
		<span bind:this={settingsButton} class="control nav-item">
			<svg><use href="/buttons.svg#settings"/></svg>
		</span>
		<!-- Fullscreen Button -->
		<span class="control nav-item" title="Fullscreen" on:click={toggleFullscreen}>
			<svg><use href="/buttons.svg#fullscreen"/></svg>
		</span>
		<!-- Help Icon -->
		<span class="control nav-item" title="Help" on:click={() => showingHelp = true}>
			<svg><use href="/buttons.svg#help"/></svg>
		</span>
	</header>

	{#if $selectedTab === 'compare'}
		<ComparisonView />
	{:else}
		<GraphView />
	{/if}
</div>

<div bind:this={popover} id="popover">
    <ViewSettings/>
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

	header {
        border-bottom: rgb(55, 59, 65) solid 1px;
        box-shadow: 0rem 0.3rem 0.3rem var(--shadow-color);

        align-items: center;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        vertical-align: middle;
        
        text-align: center;
        z-index: 999;        /* Make sure nav bar is on top */
    }
    /* Since we use SVG buttons we need to specify a nonzero size */
    .control svg {
        height: 1.5rem;
        width: 1.5rem;
    }
    .nav-item {
        display: block;
        font-size: 1.25rem;
        margin: 1rem 0.5rem;
    }
    .separator {
        border-left: 1px solid var(--nav-gray);
        height: 1.5rem;
        margin: 0 0.5rem 0 0.5rem;
    }
    .spacer {
        flex: 1;
    }
    .tab {
        margin: 1rem 0.5rem;
    }
    .tab-container {
        border: 1px solid var(--nav-gray);
        border-radius: 0.5rem;
        padding: 0.5rem 1rem 0.5rem 1rem;
        margin: 0.5rem;
    }
    .tab-selected {
        font-weight: bold;
    }
</style>