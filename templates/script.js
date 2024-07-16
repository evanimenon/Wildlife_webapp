document.getElementById('scroll-to-upload').addEventListener('click', function(event) {
    event.preventDefault();
    document.querySelector('#upload').scrollIntoView({ 
        behavior: 'smooth' 
    });
});
