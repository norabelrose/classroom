import type { Core, Stylesheet } from "cytoscape";


const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');
const contrastColor = themeQuery.matches ? 'white' : 'black';


export const graphStyles: Stylesheet[] = [
    { selector: '*', style: { color: contrastColor } },
    {
        selector: 'node',
        style: {
            'background-color': '#1f78b4',  // Light blue; same color as NetworkX plots
            'background-fit': 'contain',
            'background-image': elem => `/thumbnail/${elem.data('id')}`,
            'label': 'data(id)',
            'shape': 'round-rectangle',
            'height': '128px',
            'width': '128px',
        }
    },
    {
        selector: 'node:selected',
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
        selector: 'node:unselected',
        style: {
            'transition-property': 'height, width',
            'transition-duration': '0.2s' as any, // Cytoscape types seem to be wrong here
            'z-index': 0,
        }
    },
    {
        selector: 'edge',
        style: {
            'curve-style': 'bezier',
            'line-color': contrastColor,
        }
    },
    {
        selector: 'edge[strict]',
        style: {
            'line-style': 'solid',
            'target-arrow-color': contrastColor,
            'target-arrow-shape': 'triangle',
        } 
    },
    {
        selector: 'edge[indiff]',
        style: {
            'line-style': 'dashed'
        }
    },
    {
        selector: 'edge:selected',
        style: {
            'line-color': 'plum',
            'target-arrow-color': 'plum',
        }
    },
];

// Register the graph to have its color scheme automatically updated to match the browser/system preference
// Returns a function that can be called to remove the listener
export function autoUpdateColorScheme(graph: Core): () => void {
    const updateTheme = (e: MediaQueryListEvent) => {
        const contrastColor = e.matches ? 'white' : 'black';
        if (graph == null)  // This could happen if the color theme is changed before the graph is loaded
            return;
        
        graph.style()   // @ts-expect-error
            .selector('*').style({ color: contrastColor })
            .selector('edge').style({ 'line-color': contrastColor })
            .selector('edge[strict]').style({ 'target-arrow-color': contrastColor })
            .update();
    };
    themeQuery.addEventListener('change', updateTheme);

    return () => themeQuery.removeEventListener('change', updateTheme);
}