import type { Core, NodeSingular } from 'cytoscape';


// 'Attach' an floating HTML element to a Cytoscape node, ensuring that its position
// keeps in sync with the node's position.
export function attachFloaterToNode(graph: Core, attached: HTMLElement, node: NodeSingular, offset = {x: 0, y: 0}): () => void {
    function update() {
        const { x: centerX, y: centerY } = node.renderedPosition();
        const width = node.renderedOuterWidth();
        const height = node.renderedOuterHeight();
        const x = centerX - width / 2;
        const y = centerY - height / 2;

        Object.assign(attached.style, {
            height: `${height}px`,
            width: `${width}px`,
            left: `${x + offset.x}px`,
            top: `${y + offset.y}px`,
        });
    }

    Object.assign(attached.style, {
        position: 'absolute',
        'z-index': 1000,
    });
    update();
    graph.container()!.appendChild(attached);

    // Now add a listener to update the position whenever the node is moved
    graph.on('position', 'node', update);
    graph.on('viewport', update);

    return () => {
        graph.removeListener('position', 'node', update);
        graph.removeListener('viewport', update);
        graph.container()!.removeChild(attached);
    };
}

type FloaterCreator = (n: NodeSingular) => HTMLElement;

export function attachFloaterOnSelect(graph: Core, elemCreator: FloaterCreator): () => void {
    let detach: () => void = () => undefined;

    graph.on('select', 'node', (event) => {
        const floater = elemCreator(event.target);
        setTimeout(() => {
            detach = attachFloaterToNode(graph, floater, event.target);
        }, 200);
    });
    graph.on('unselect', 'node', _ => {
        detach();
        detach = () => undefined;
    });
    return detach;
}