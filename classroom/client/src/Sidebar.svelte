<script lang="ts">
    export let open = false;
</script>

<aside class="absolute w-full h-full bg-gray-200 border-r-2 shadow-lg sm:hidden" class:open>
    <nav class="p-12 text-xl">
        <a class="block" href="#about">About</a>
        <a class="block" href="#contact">Contact</a>
    </nav>
</aside>

<style>
    aside {
        /* offscreen by default */
        left: -100%;
        transition: left 0.3s ease-in-out
    }
    .open {
        /* slide on screen */
        left: 0
    }
</style>
