const circles = 
document.querySelectorAll('.circle');
let activeLight = 2;

setInterval(changeLight, 10000);

function changeLight(){
    circles[activeLight].className = 'circle';
    activeLight--;

    if(activeLight < 0){
        activeLight = 2;
    }

    const currentLight = circles[activeLight];

    currentLight.classList.add(currentLight.getAttribute('color'));
}