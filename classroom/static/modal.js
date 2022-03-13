function openModal(name) {
    // The modal background appears with no animation- can we make it fade in?
    const modal = document.getElementById(name + '-modal');
    const window = document.getElementById(name + '-window');
    const ESC_KEYS = ['Escape', 'Enter'];

    // Allow clicking on the background to close the modal
    modal.addEventListener('click', function callback() {
        closeModal(name);
        modal.removeEventListener('click', callback);
    });
    // But not inside the window itself
    window.addEventListener('click', (e) => e.stopPropagation());

    // Allow pressing the escape or enter/return keys to close the modal
    document.addEventListener('keydown', function callback(e) {
        if (ESC_KEYS.includes(e.key)) {
            closeModal(name);
            document.removeEventListener('keydown', callback);
        }
    });
    modal.style.display = 'block';
}
function closeModal(name) {
    const modal = document.getElementById(name + '-modal');
    const window = document.getElementById(name + '-window');

    // Set display to none after animation is complete
    window.addEventListener('animationend', function callback() {
        window.classList.remove('modal-pop-out');
        modal.style.display = 'none';

        // Remove the event listener after the animation is complete
        window.removeEventListener('animationend', callback);
    })
    window.classList.add('modal-pop-out');
}