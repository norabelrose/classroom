<script lang="ts">
    import { scale } from 'svelte/transition';
    import { onDestroy, onMount } from 'svelte';
    import { autoUpdate, arrow, computePosition, offset, shift } from '@floating-ui/dom';

    export let anchorId: string;
    export let close: () => void;
    let arrowElem: HTMLElement;
    let popover: HTMLElement;
    
    const anchor = document.getElementById(anchorId)!;
    if (anchor == null) throw new Error(`No anchor element with id ${anchorId}`);

    let cleanupFn: () => void;
    onMount(async () => {
        const updatePos = async () => {
            const {x, y, middlewareData} = await computePosition(anchor, popover, {
                middleware: [
                    offset(16),             // 16px distance from the anchor
                    shift({ padding: 8 }),  // Make sure popover is at least 8px from edge of the screen
                    arrow({ element: arrowElem })   // Keep the arrow pointing at the anchor
                ],
                placement: 'bottom'
            });
            const arrowData = middlewareData.arrow;
            const arrowX = arrowData?.x;
            const arrowY = arrowData?.y;

            arrowElem.style.left = arrowX != null ? `${arrowX}px` : '';
            arrowElem.style.top = arrowY != null ? `${arrowY}px` : '';
            popover.style.left = `${x}px`;
            popover.style.top = `${y}px`;
        };
        await updatePos();

        cleanupFn = autoUpdate(anchor, popover, updatePos);
    });
    onDestroy(() => { if (cleanupFn) cleanupFn(); });
</script>

<div class="fullscreen background" on:click={close}/>
<div bind:this={arrowElem} id="arrow"/>
<div bind:this={popover} class="popover" transition:scale={{ duration: 200 }}>
    <div class="popover-content">
        <slot name="content"/>
    </div>
</div>

<style>
    #arrow {
        background: #333;
        width: 8px;
        height: 8px;
        position: absolute;
        transform: rotate(45deg);
    }
    .background {
        background-color: transparent;
        height: 100%;
        position: absolute;
        width: 100%;
        z-index: 999;
    }
    .popover {
        background-color: var(--nav-color);
        border: rgb(55, 59, 65) solid 1px;
        border-radius: 0.5rem;
        box-shadow: 0rem 0.3rem 0.3rem var(--shadow-color);
        padding: 1rem;
        position: absolute;
        user-select: none;
        z-index: 1000;
    }
</style>