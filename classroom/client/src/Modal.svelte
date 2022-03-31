<script lang="ts">
    import { fade, fly } from 'svelte/transition';

	export let visible: boolean;
    const close = () => visible = false;
</script>

<svelte:window on:keydown={(e) => {
    if (e.key === 'Escape') close();
}}/>

{#if visible}
    <div class="fullscreen container">
        <div class="fullscreen background" on:click={close} transition:fade={{ duration: 200 }}/>
        <div class="window" role="dialog" transition:fly={{ y: 200, duration: 200 }}>
            <div class="close control" on:click={close}>
                &times;
            </div>
            <slot name="header"/>
            <hr/>
            <div class="content"><slot name="content"/></div>
        </div>
    </div>
{/if}

<style>
    .background {
        background: rgba(0,0,0,0.4);
    }
    .close {
        float: right;
        font-size: 1.5rem;
        position: fixed;
    }
    .content {
        text-align: left;
    }
	.container {
        align-items: center;
        display: flex;
        justify-content: center;
	}
	.window {
        background-color: var(--bg-color);
        border-radius: 1rem;
        border: 1px solid #888;
        box-shadow: 0rem 0.5rem 0.5rem var(--shadow-color);
        max-height: calc(100vh - 4em);
		max-width: 32em;
        overflow-x: clip;
		overflow-y: auto;
        padding: 1.5rem;
        text-align: center;
        transform: translateY(-25vh);
        width: 50%;
        z-index: 1000;
	}
</style>