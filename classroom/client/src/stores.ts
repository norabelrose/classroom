import { writable, Writable } from 'svelte/store';


// Wrapper around a writable store that syncs the value with localStorage
export function persistent<T>(key: string, defaultValue: T): Writable<T> {
    // If we have a value in localStorage, use that instead of the default
    const oldValue = localStorage.getItem(key);
    const initialValue = oldValue ? JSON.parse(oldValue) : defaultValue;

    const {subscribe, set, update} = writable(initialValue, (set) => {
        // Sign up for localStorage changes when we get our first subscriber
        const storageDidUpdate = (event: StorageEvent) => {
            if (event.key === key)
                set(event.newValue ? JSON.parse(event.newValue) : defaultValue);
        };
        window.addEventListener("storage", storageDidUpdate);
        return () => window.removeEventListener("storage", storageDidUpdate);
    });

    return {
        subscribe, update,
        set(value: T) {
            localStorage.setItem(key, JSON.stringify(value));
            set(value);
        },
    };
}


type RewardType = 'borda' | 'bradley-terry' | 'thurstone' | 'none';
type Tab = 'compare' | 'visualize';

export const rewardType = persistent<RewardType>('rewardType', 'bradley-terry');
export const selectedTab = persistent<Tab>('selectedTab', 'compare');
export const showEnvRewards = persistent('showEnvRewards', false);
export const showIndifferences = persistent('showIndifferences', true);
export const showRedundant = persistent('showRedundant', true);


// This will be null when there is only one run we are allowed to see
export const selectedRun: Writable<string | null> = writable(null);
