// SearchForm Section code
let searchForm = document.querySelector('.search-form');
document.querySelector('#search-btn').onclick = () =>{
    searchForm.classList.toggle('active');
}
window.onscroll = () =>{
    searchForm.classList.remove('active');
}

$(document).ready(function(){

    $(window).scroll(function(){
        //Navbar-bottom scrolling
        if(this.scrollY > 5){
            $('.navbar-bottom').addClass("sticky");
        }else{
            $('.navbar-bottom').removeClass("sticky");
        }

        //Scrolling Button Btn
        if(this.scrollY > 500){
            $('.scroll-up-btn').addClass("show");
        }else{
            $('.scroll-up-btn').removeClass("show");
        }
    });

        //slide up script
        $('.scroll-up-btn').click(function(){
            $('html').animate({scrollTop:0});
        });
        
    //owl owlCarousel
    $('.owl-carousel').owlCarousel({
        margin: 5,
        mavigation: true,
        loop: true,
        autoplay: true,
        autoplayTimeOut: 2000,
        autoplayHoverPause: true,

        responsive: {
            0: {
                items: 1,
                nav: false
            },
            600: {
                items: 2,
                nav: false
            },
            1000: {
                items: 3,
                nav: false
            }
        }
    });
<<<<<<< HEAD
});




document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("contactForm");

    form.addEventListener("submit", function (e) {
        e.preventDefault(); // Prevent default form submission
        console.log("Form submission prevented");

        const formData = new FormData(form); // Collect form data

        fetch("/submit_contact/", {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": getCookie("csrftoken"), // CSRF token for Django
            },
        })
        .then(response => {
            if (!response.ok) {
                console.error("Network response was not ok.");
                return response.json().then(data => {
                    console.error("Error data:", data);
                    alert("Error: " + (data.error || "Submission failed"));
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.message) {
                alert(data.message); // Show success message
            } else if (data.errors) {
                alert("There was an error with your submission.");
                console.error("Form errors:", data.errors);
            }
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== "") {
            const cookies = document.cookie.split(";");
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === name + "=") {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});

=======
});
>>>>>>> edd1b94cd0bc1b6dbd76581cfb87a6ec43e96d1a
