import type { NodeSingular, Stylesheet } from "cytoscape";
import { showIndifferences } from "../stores";
import { get } from "svelte/store";


const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');
const contrastColor = themeQuery.matches ? 'white' : 'black';


export const graphStyles: Stylesheet[] = [
    { selector: '*', style: { color: contrastColor } },
    {
        selector: 'node.clip',
        style: {
            'background-color': '#1f78b4',  // Light blue; same color as NetworkX plots
            'background-fit': 'contain',
            'background-image': (elem: NodeSingular) => `/thumbnail/${elem.data('id')}/60`,
            'label': (elem: NodeSingular) => {
                const reward: number = elem.data('reward');
                return reward !== undefined ? reward.toLocaleString() : '';
            },
            'shape': 'round-rectangle',
            'height': '128px',
            'width': '128px',
        }
    },
    {
        selector: 'node.clip:selected',
        style: {
            'border-width': '4px',
            'border-color': 'plum',
            'height': '480px',
            'width': '480px',
            'transition-property': 'height, width',
            'transition-duration': '0.2s' as any, // Cytoscape types seem to be wrong here
            'transition-timing-function': 'spring(0.5, 0.5)' as any,
            'z-index': 100, // Selected nodes should be on top of unselected nodes
        }
    },
    {
        selector: 'node.clip:unselected',
        style: {
            'transition-property': 'height, width',
            'transition-duration': '0.2s' as any, // Cytoscape types seem to be wrong here
            'z-index': 0,
        }
    },
    {
        selector: 'edge.pref, .eh-preview',
        style: {
            'curve-style': 'bezier',
            'line-color': contrastColor,
        }
    },
    {
        selector: 'edge[?strict], .eh-ghost-edge',
        style: {
            'arrow-scale': 2,
            'line-style': 'solid',
            'target-arrow-color': contrastColor,
            'target-arrow-shape': 'triangle',
        } 
    },
    {
        selector: 'edge.pref[!strict]',
        style: {
            'line-style': 'dashed',
            'visibility': get(showIndifferences) ? 'visible' : 'hidden',
        }
    },
    {
        selector: 'edge.pref:selected',
        style: {
            'line-color': 'plum',
            'target-arrow-color': 'plum',
        }
    },
    {
        // Green plus button handle for adding edges
        selector: '.eh-handle',
        style: {
            'background-color': 'green',
            'content': "+",
            'text-valign': 'center',
        }
    },
    {
        selector: '.eh-source, .eh-presumptive-target',
        style: {
            'border-color': 'green',
            'border-width': '4px',
        }
    },
    {
        // Hide the ghost edge when a preview is being displayed
        selector: 'edge.eh-preview-active',
        style: { 'opacity': 0 }
      }
];
