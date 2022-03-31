import { readable, writable, Writable } from 'svelte/store';


export const availableRuns = readable([], set => {
    fetch('/available_runs').then(res => res.json()).then(set).catch(console.error);
});

// This will be null when there is only one run we are allowed to see
export const selectedRun: Writable<string | null> = writable(null);
