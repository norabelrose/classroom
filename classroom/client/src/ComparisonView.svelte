<script lang="ts">
    export let clipA = 0;
    export let clipB = 1;

    let socket = new WebSocket(`ws://${location.host}/feedback`);
    socket.addEventListener('open', function (event) {
        socket.send('Hello Server!');
    });

    function indifferent() {
        clipA = Math.floor(Math.random() * 3000);
        clipB = Math.floor(Math.random() * 3000);
    }
</script>

<div id="container">
    <div>
        <div class="clips">
            <div>
                <h2>Clip A</h2>
                <iframe title="Clip A" src={`/viewer_html/${clipA}`}/>
            </div>
            <div>
                <h2>Clip B</h2>
                <iframe title="Clip B" src={`/viewer_html/${clipB}`}/>
            </div>
        </div>
        <div class="buttons">
            <button on:click={() => clipA = Math.floor(Math.random() * 3000)}>
                Clip A is better
            </button>
            <button on:click={() => socket.send("Indifferent")}>
                Too close to tell
            </button>
            <button on:click={() => clipB = Math.floor(Math.random() * 3000)}>
                Clip B is better
            </button>
        </div>
        <div class="buttons">
            <button>
                Skip
            </button>
        </div>
    </div>
</div>

<style>
    #container {
        align-items: center;
        display: flex;
        flex-direction: column;
        height: 100%;
        width: 100%;
    }
    .clips {
        display: flex;
        flex-direction: row;
    }
    .buttons {
        display: flex;
        justify-content: space-around;
        width: 100%;
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