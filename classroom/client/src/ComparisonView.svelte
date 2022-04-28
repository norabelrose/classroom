<script lang="ts">
    import { GraphController, Pref, QueryPair } from './graph/graph_controller';
    import { Jumper } from 'svelte-loading-spinners';
    import { onMount } from 'svelte';

    let query: QueryPair;
    onMount(async () => query = await GraphController.currentPair());

    const key2pref: Record<string, Pref> = {
        // WASD layout
        'a': '>',
        's': '=',
        'd': '<',

        // Arrow keys
        'ArrowLeft': '>',
        'ArrowRight': '<',
        'ArrowUp': '=',
    };

    // We support WASD & arrow keybindings for indicating preferences. In order to make
    // sure the user is aware of which clip they're selecting, we show the corresponding
    // mathematical symbol (>, <, or =) in between the two clips, and make the preferred
    // clip momentarily larger than the unfavored one. To make sure that the animations
    // don't slow down more experienced users, we tie the duration of this visual feedback
    // to the length of time that the key is pressed down.
    // This creates an annoying problem, though- what if the user presses down on another
    // key before releasing the first one? Which preference should get committed to the
    // database? I've decided that the LAST key released should win, since this allows
    // users to 'undo' their initial keypress once they see the visual feedback by simply
    // pressing down on a different key. We store the currently pressed keybindings in a
    // stack to implement this behavior.
    let prefStack: Pref[] = [];
    // Automagically synced w/ the last item in stack. Would be nice to use .at() here
    // but it's not supported by iOS WebKit until iOS 15.4
    $: highlight = prefStack[prefStack.length - 1] ?? null;

    function handleKeyDown(e: KeyboardEvent) {
        if (e.repeat) return;

        // Escape 'unstages' the currently selected preference and resets the stack
        if (e.key === 'Escape') {
            prefStack = [];
            return;
        }

        const pref = key2pref[e.key];
        if (!pref) return;

        prefStack = [...prefStack, pref];
    }
    function handleKeyUp(e: KeyboardEvent) {
        const pref = key2pref[e.key];
        if (prefStack.includes(pref)) releasePref(pref);
    }
    function releasePref(pref: Pref) {
        prefStack = prefStack.filter(p => p !== pref);

        // Actually commit the preference to the server
        if (!prefStack.length) {
            GraphController.commitFeedback(pref, query).then(msg => query = msg);
        }
    }
</script>

<svelte:window on:keydown={handleKeyDown} on:keyup={handleKeyUp} />

<div id="container">
    {#if !query}
        <Jumper />
    {:else}
        <div>
            <div class="clips">
                <div class:magnified={highlight === '>'} class:minified={highlight && highlight !== '>'} id="gt">
                    <h2>Left</h2>
                    <iframe title="Left" src={`/viewer_html/${query.left}`}/>
                </div>
                <div id="symbol">{highlight ?? ' '}</div>
                <div class:magnified={highlight === '<'} class:minified={highlight && highlight !== '<'} id="lt">
                    <h2>Right</h2>
                    <iframe title="Right" src={`/viewer_html/${query.right}`}/>
                </div>
            </div>
            <div class="buttons">
                <button
                    on:mousedown={() => prefStack = [...prefStack, '>']}
                    on:mouseup={() => releasePref('>')}
                    on:mouseleave={() => prefStack = []}>
                    Left is better
                </button>
                <button
                    on:mousedown={() => prefStack = [...prefStack, '=']}
                    on:mouseup={() => releasePref('=')}
                    on:mouseleave={() => prefStack = []}>
                    Too close to tell
                </button>
                <button
                    on:mousedown={() => prefStack = [...prefStack, '<']}
                    on:mouseup={() => releasePref('<')}
                    on:mouseleave={() => prefStack = []}>
                    Right is better
                </button>
            </div>
        </div>
    {/if}
</div>

<style>
    #container {
        align-items: center;
        display: flex;
        flex-direction: column;
        height: 100%;
        width: 100%;
    }
    #symbol {
        align-self: center;
        color: var(--font-color);
        font-size: 3em;
        font-weight: bold;
        margin: 0.5rem;
        text-align: center;
        width: 2rem;
    }
    .buttons {
        display: flex;
        justify-content: space-around;
        width: 100%;
    }
    .clips {
        display: flex;
        flex-direction: row;
    }
    .magnified {
        transition: 0.2s;
    }
    #gt.magnified {
        transform: scale(1.1) translateX(-5%);
    }
    #lt.magnified {
        transform: scale(1.1) translateX(5%);
    }
    .minified {
        opacity: 0.9;
        transform: scale(0.9);
        transition: 0.2s;
    }
    button {
        margin: 1rem;
    }
    h2 {
        text-align: center;
    }
    iframe {
        border: none;
        margin: 1rem;

        height: 480px;
        width: 480px;
    }
</style>