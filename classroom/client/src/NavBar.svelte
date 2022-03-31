<script lang="ts">
    import HamburgerButton from './HamburgerButton.svelte';
    import Popover from './Popover.svelte';
    import ViewSettings from './ViewSettings.svelte';
    import { availableRuns, selectedRun } from './stores';

    export let showingHelp = false;
    export let showingSettings = false;
    export let showingSidebar = false;
    export let showingRuns = false;
    
    // For the fullscreen button in the nav bar
	function toggleFullscreen() {
		if (document.fullscreenElement) {
			document.exitFullscreen();
		} else {
			document.documentElement.requestFullscreen();
		}
	}

    const KEYBINDINGS: Record<string, Function> = {
        f: toggleFullscreen,
        h: () => showingHelp = !showingHelp,
        // r: () => showingRuns = !showingRuns,
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

<header class="nav">
    <HamburgerButton bind:open={showingSidebar}/>
	<!-- Undo Button -->
	<span class="control nav-item" id="undo" title="Undo">
		<svg><use href="/buttons.svg#undo"/></svg>
	</span>
	<span class="spacer"></span>
	<!-- Run Selector; disabled when there are no (other) available runs -->
    <span class="run-selector"  class:control={$availableRuns.length > 1}
                                on:click={() => showingRuns = $availableRuns.length > 1}>
		{$selectedRun ?? "Classroom"}
	</span>
    <span class="spacer"></span>
	<!-- Fullscreen Button -->
	<span class="control nav-item" title="Fullscreen" on:click={toggleFullscreen}>
		<svg><use href="/buttons.svg#fullscreen"/></svg>
	</span>
	<!-- Settings Icon -->
    <span class="control nav-item" id="settings-icon" on:click={() => showingSettings = true}>
        <svg><use href="/buttons.svg#settings"/></svg>
    </span>
	<!-- Help Icon -->
	<span class="control nav-item" title="Help" on:click={() => showingHelp = true}>
		<svg><use href="/buttons.svg#help"/></svg>
	</span>
</header>

{#if showingSettings}
    <Popover anchorId="settings-icon" close={() => showingSettings = false}>
        <ViewSettings slot="content"/>
    </Popover>
{/if}

<style>
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
        font-weight: bold;
        margin: 1rem 0.5rem;
    }
    .run-selector {
        border: var(--nav-gray) solid 1px;
        border-radius: 0.5rem;
        color: var(--nav-gray);
        font-size: 1rem;
        padding: 0.5rem 1rem 0.5rem 1rem;
        margin: 0.5rem;
        user-select: none;
    }
    .spacer {
        flex: 1;
    }
</style>